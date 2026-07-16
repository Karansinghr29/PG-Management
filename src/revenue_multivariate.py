"""Multivariate revenue forecasting: revenue history + historical occupancy.

Manager requirement: next-month revenue should depend not only on past revenue
but also on historical occupancy (from the real booking/stay dataset).

This module reuses `revenue_ml._build_frame` for the revenue-lag base and adds
LAGGED occupancy drivers from `property_month` (occupancy %, active tenants,
avg rental, new bookings, move-outs, notices — all shifted, so known before
month t). It compares the existing revenue-only model against the revenue +
occupancy multivariate model on identical chronological walk-forward windows,
computes the occupancy↔revenue relationship, feature importance (+ SHAP), and a
±5% occupancy scenario. No synthetic data, no leakage.

Run:  python -m src.revenue_multivariate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import revenue_ml  # noqa: E402

TARGET = "revenue"
# Lagged occupancy drivers (all known before month t -> leakage-safe).
OCC_FEATURES = ["occupancy_pct_lag1", "active_tenants_occ_lag1",
                "avg_monthly_rental_lag1", "new_bookings_lag1",
                "move_outs_lag1", "move_ins_lag1", "notice_count_lag1"]
REV_FEATURES = revenue_ml.FEATURES                    # revenue-only baseline
MULTI_FEATURES = REV_FEATURES + OCC_FEATURES
N_TEST = 12


def _frame(pm):
    """Revenue-lag base (reused) + occupancy lag columns (already in pm)."""
    df = revenue_ml._build_frame(pm)
    for c in OCC_FEATURES:
        df[c] = pm.sort_values("billing_period").reset_index(drop=True)[c]
    return df


def _models():
    # Linear models are scaled (many features on very different scales) and Ridge
    # is regularised to handle the collinearity between revenue lags and occupancy.
    m = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            random_state=config.RANDOM_STATE, verbosity=0)
    except ImportError:
        pass
    return m


def _walk_forward(df, feats, months, usable, test_idx):
    """One-step expanding-window predictions per model for one feature set."""
    models = _models()
    preds = {name: [] for name in models}
    for i in test_idx:
        tr = [u for u in usable if u < i]
        X_tr = df.loc[tr, feats].to_numpy(float)
        y_tr = df.loc[tr, TARGET].to_numpy(float)
        X_te = df.loc[[i], feats].to_numpy(float)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds[name].append(float(model.predict(X_te)[0]))
    return preds


def run():
    t0 = time.time()
    pm = fe.build_all()["property_month"]
    df = _frame(pm)
    months = df["billing_period"].astype(str).to_numpy()

    usable = df.dropna(subset=MULTI_FEATURES).index.to_numpy()
    test_idx = usable[usable >= len(df) - N_TEST]
    test_idx = [i for i in test_idx if len([u for u in usable if u < i]) >= 6]
    actuals = df.loc[test_idx, TARGET].to_numpy(float)
    test_months = [months[i] for i in test_idx]

    rev_preds = _walk_forward(df, REV_FEATURES, months, usable, test_idx)
    multi_preds = _walk_forward(df, MULTI_FEATURES, months, usable, test_idx)

    rows = []
    for tag, preds in [("RevenueOnly", rev_preds), ("Revenue+Occupancy", multi_preds)]:
        for name, p in preds.items():
            m = utils.regression_metrics(actuals, np.array(p))
            m["FeatureSet"], m["Model"] = tag, name
            rows.append(m)
    comp = pd.DataFrame(rows)[["FeatureSet", "Model", "MAE", "RMSE", "MAPE", "R2"]]
    comp = comp.round(2).sort_values("MAPE").reset_index(drop=True)
    comp.to_csv(config.OUT_DIR / "comparison_multivariate.csv", index=False)

    best_rev = comp[comp.FeatureSet == "RevenueOnly"].iloc[0]
    best_multi = comp[comp.FeatureSet == "Revenue+Occupancy"].iloc[0]
    pd.DataFrame({
        "billing_period": test_months, "actual": actuals,
        "revenue_only": rev_preds[best_rev.Model],
        "multivariate": multi_preds[best_multi.Model],
    }).to_csv(config.OUT_DIR / "backtest_multivariate.csv", index=False)

    # ---- Occupancy <-> revenue relationship (real) --------------------------- #
    valid = pm[["occupancy_pct", "revenue"]].dropna()
    r, pval = stats.pearsonr(valid["occupancy_pct"], valid["revenue"])
    slope, intercept, *_ = stats.linregress(valid["occupancy_pct"],
                                            valid["revenue"])
    valid.assign(reg_line=intercept + slope * valid["occupancy_pct"]) \
         .to_csv(config.OUT_DIR / "occupancy_revenue_scatter.csv", index=False)

    # ---- Final multivariate model + importance + scenario -------------------- #
    best_model_name = best_multi.Model
    final = _models()[best_model_name]
    Xall = df.loc[usable, MULTI_FEATURES].to_numpy(float)
    yall = df.loc[usable, TARGET].to_numpy(float)
    final.fit(Xall, yall)
    _feature_importance(final, df.loc[usable, MULTI_FEATURES], yall,
                        best_model_name)
    scenario = _scenario(df, final)
    future = _future_forecast(df, final, steps=6)
    future.to_csv(config.OUT_DIR / "forecast_multivariate.csv", index=False)

    import joblib
    joblib.dump(final, config.MODEL_DIR / "revenue_multivariate_best.pkl")

    winner = ("Revenue+Occupancy" if best_multi.MAPE < best_rev.MAPE
              else "RevenueOnly")
    meta = {
        "revenue_only_best": {"model": best_rev.Model, "mape": float(best_rev.MAPE),
                              "mae": float(best_rev.MAE), "rmse": float(best_rev.RMSE),
                              "r2": float(best_rev.R2)},
        "multivariate_best": {"model": best_multi.Model,
                              "mape": float(best_multi.MAPE),
                              "mae": float(best_multi.MAE),
                              "rmse": float(best_multi.RMSE),
                              "r2": float(best_multi.R2)},
        "winner": winner,
        "improvement_mape_pct_points": round(float(best_rev.MAPE - best_multi.MAPE), 2),
        "occupancy_revenue_corr": round(float(r), 3),
        "corr_pvalue": float(pval),
        "regression_slope_rev_per_occ_pct": round(float(slope), 1),
        "features": MULTI_FEATURES,
        "occupancy_features": OCC_FEATURES,
        "n_test_months": int(len(actuals)),
        "n_train_months_total": int(len(usable)),
        "validation": "chronological expanding-window walk-forward (one-step)",
        "scenario": scenario,
        "next_month_revenue": float(future["revenue"].iloc[0]),
        "trained_at": revenue_ml._utc_now(),
        "training_duration_sec": round(time.time() - t0, 1),
        "model_version": revenue_ml._utc_version(),
    }
    (config.OUT_DIR / "model_meta_multivariate.json").write_text(
        json.dumps(meta, indent=2))

    _plot(test_months, actuals, rev_preds[best_rev.Model],
          multi_preds[best_multi.Model], best_rev.Model, best_multi.Model,
          valid, slope, intercept)

    print(f"Occupancy<->revenue Pearson r = {r:.3f} (p={pval:.2e})")
    print(comp.to_string(index=False))
    print(f"\nWinner: {winner} | revenue-only MAPE {best_rev.MAPE} -> "
          f"multivariate MAPE {best_multi.MAPE}")
    print("Scenario:", scenario)
    return comp


def predict_live(pm=None, model_name: str = "Ridge", steps: int = 6) -> dict:
    """Live multivariate revenue prediction — NO file writes, NO plots.

    Single source of truth for the Ridge multivariate next-month revenue shown on
    the Revenue Forecast KPI card, AI Recommendations and Financial Overview.
    Fixed to the Ridge multivariate model (Option A). Uses the exact same frame,
    walk-forward windows and recursive future-forecast logic as ``run`` so the
    number matches the persisted model output, but is recomputed live from the
    passed ``property_month`` (never reads model_meta_multivariate.json).

    The headline ``next_month_revenue`` (first future step) depends only on the
    live property_month (its occupancy lag is the last observed month), so it
    tracks the data automatically. Returns model, walk-forward MAE/RMSE/MAPE,
    next_month_revenue, ±5% scenario, occupancy↔revenue correlation and the
    6-month future frame.
    """
    if pm is None:
        pm = fe.build_all()["property_month"]
    df = _frame(pm)
    usable = df.dropna(subset=MULTI_FEATURES).index.to_numpy()
    test_idx = usable[usable >= len(df) - N_TEST]
    test_idx = [i for i in test_idx if len([u for u in usable if u < i]) >= 6]
    actuals = df.loc[test_idx, TARGET].to_numpy(float)

    # Ridge-only walk-forward metrics (same expanding windows as run()).
    preds = []
    for i in test_idx:
        tr = [u for u in usable if u < i]
        mdl = _models()[model_name]
        mdl.fit(df.loc[tr, MULTI_FEATURES].to_numpy(float),
                df.loc[tr, TARGET].to_numpy(float))
        preds.append(float(mdl.predict(df.loc[[i], MULTI_FEATURES].to_numpy(float))[0]))
    if len(actuals):
        m = utils.regression_metrics(actuals, np.array(preds))
    else:
        m = {"MAE": float("nan"), "RMSE": float("nan"),
             "MAPE": float("nan"), "R2": float("nan")}

    final = _models()[model_name]
    final.fit(df.loc[usable, MULTI_FEATURES].to_numpy(float),
              df.loc[usable, TARGET].to_numpy(float))
    future = _future_forecast(df, final, steps=steps)

    valid = pm[["occupancy_pct", "revenue"]].dropna()
    corr = (float(stats.pearsonr(valid["occupancy_pct"], valid["revenue"])[0])
            if len(valid) > 1 else float("nan"))

    return {
        "model": model_name,
        "mape": round(float(m["MAPE"]), 2),
        "mae": float(m["MAE"]),
        "rmse": float(m["RMSE"]),
        "next_month_revenue": float(future["revenue"].iloc[0]),
        "scenario": _scenario(df, final),
        "occupancy_revenue_corr": round(corr, 3),
        "future": future,
    }


def _feature_importance(model, X, y, model_name):
    from sklearn.inspection import permutation_importance
    cols = list(X.columns)
    r = permutation_importance(model, X.to_numpy(float), y, n_repeats=10,
                               random_state=config.RANDOM_STATE, scoring="r2")
    (pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)
       .to_csv(config.OUT_DIR / "perm_importance_multivariate.csv"))
    # SHAP requires a tree model. If the best model is not tree-based, fit a
    # RandomForest on the same features purely for the SHAP importance view.
    try:
        import shap
        tree = model if hasattr(model, "feature_importances_") else \
            RandomForestRegressor(n_estimators=300,
                                  random_state=config.RANDOM_STATE).fit(
                X.to_numpy(float), y)
        vals = shap.TreeExplainer(tree).shap_values(X.to_numpy(float))
        mean_abs = np.abs(vals).mean(axis=0)
        (pd.Series(mean_abs, index=cols).sort_values(ascending=False)
           .to_csv(config.OUT_DIR / "shap_multivariate.csv"))
    except Exception as e:
        print("  (SHAP skipped:", e, ")")


def _future_forecast(df, model, steps: int = 6) -> pd.DataFrame:
    """Recursive multi-step future revenue forecast from the multivariate model.

    Revenue lags are fed recursively from the model's own predictions. Occupancy
    lags use the SEPARATELY-forecast occupancy series (forecast_occupancy_pct /
    forecast_active_tenants, produced from past occupancy only) so there is no
    future-occupancy leakage; the remaining flow drivers (move-ins/outs, notices,
    new bookings, avg rental) use recent-3-month persistence.
    """
    rev_hist = df[TARGET].tolist()                 # observed revenue history
    last = df.iloc[-1]
    recent = df.tail(3)
    # Persistence values for flow drivers (recent mean).
    persist = {c.replace("_lag1", ""): float(recent[c.replace("_lag1", "")].mean())
               for c in ["avg_monthly_rental_lag1", "new_bookings_lag1",
                         "move_outs_lag1", "move_ins_lag1", "notice_count_lag1"]}
    # Forecast occupancy inputs (past-only forecasts already on disk).
    occ_fc = _read_forecast("forecast_occupancy_pct.csv", "occupancy_pct")
    ten_fc = _read_forecast("forecast_active_tenants.csv", "active_tenants")

    last_period = df["billing_period"].iloc[-1]
    if not isinstance(last_period, pd.Period):
        last_period = pd.Period(str(last_period), freq="M")
    periods = pd.period_range(last_period + 1, periods=steps, freq="M")

    occ_prev = float(last["occupancy_pct"])
    ten_prev = float(last["active_tenants_occ"])
    preds = []
    for h, p in enumerate(periods):
        feat = {
            "rev_lag1": rev_hist[-1], "rev_lag2": rev_hist[-2],
            "rev_lag3": rev_hist[-3],
            "rev_lag12": rev_hist[-12] if len(rev_hist) >= 12 else rev_hist[0],
            "rev_roll3_lag": float(np.mean(rev_hist[-3:])),
            "tenants_lag1": ten_prev, "month_num": p.month, "year": p.year,
            "occupancy_pct_lag1": occ_prev,
            "active_tenants_occ_lag1": ten_prev,
            "avg_monthly_rental_lag1": persist["avg_monthly_rental"],
            "new_bookings_lag1": persist["new_bookings"],
            "move_outs_lag1": persist["move_outs"],
            "move_ins_lag1": persist["move_ins"],
            "notice_count_lag1": persist["notice_count"],
        }
        vec = np.array([[feat[c] for c in MULTI_FEATURES]], dtype=float)
        yhat = float(model.predict(vec)[0])
        preds.append(yhat)
        rev_hist.append(yhat)
        # advance occupancy inputs from the separate occupancy forecast
        occ_prev = occ_fc[h] if h < len(occ_fc) else occ_prev
        ten_prev = ten_fc[h] if h < len(ten_fc) else ten_prev
    return pd.DataFrame({"billing_period": [str(p) for p in periods],
                         "revenue": [round(v, 1) for v in preds]})


def _read_forecast(fname, col):
    p = config.OUT_DIR / fname
    if p.exists():
        d = pd.read_csv(p)
        if col in d.columns:
            return d[col].tolist()
    return []


def _scenario(df, model):
    """±5% occupancy scenario on next month's real lag vector."""
    base = revenue_ml._next_month_features(df)[0].tolist()   # revenue-lag part
    # Next month's occupancy lag1 = the last observed month's actual value.
    last = df.iloc[-1]
    occ_vals = [float(last[c.replace("_lag1", "")]) for c in OCC_FEATURES]
    out = {}
    for label, factor in [("base", 1.0), ("plus_5pct", 1.05), ("minus_5pct", 0.95)]:
        occ = list(occ_vals)
        # scale occupancy % and active tenants (the occupancy magnitude drivers)
        for idx, name in enumerate(OCC_FEATURES):
            if name in ("occupancy_pct_lag1", "active_tenants_occ_lag1"):
                occ[idx] = occ_vals[idx] * factor
        vec = np.array([base + occ], dtype=float)
        out[label] = round(float(model.predict(vec)[0]), 1)
    return out


def _plot(months, actuals, rev_p, multi_p, rev_name, multi_name, valid, slope,
          intercept):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(months, actuals, marker="o", color="#2A9D8F", lw=2, label="actual")
    ax[0].plot(months, rev_p, marker="s", ls="--", color="#E76F51",
               label=f"Revenue-only ({rev_name})")
    ax[0].plot(months, multi_p, marker="^", ls=":", color="#264653",
               label=f"Revenue+Occupancy ({multi_name})")
    ax[0].set_title("Actual vs Predicted revenue (walk-forward)")
    ax[0].tick_params(axis="x", rotation=90, labelsize=7); ax[0].legend()
    ax[1].scatter(valid["occupancy_pct"], valid["revenue"], color="#2A9D8F",
                  alpha=0.7)
    xs = np.linspace(valid["occupancy_pct"].min(), valid["occupancy_pct"].max(), 50)
    ax[1].plot(xs, intercept + slope * xs, color="#E76F51", lw=2,
               label="regression line")
    ax[1].set_xlabel("occupancy %"); ax[1].set_ylabel("revenue")
    ax[1].set_title("Occupancy vs Revenue"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "revenue_multivariate.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
