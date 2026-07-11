"""Second occupancy forecaster: supervised ML on the real monthly panel.

This is an ADDITIONAL model that COMPLEMENTS (does not replace) the primary
Holt-Winters occupancy forecaster in `src/forecasting.py`. Holt-Winters stays
the production baseline for the multi-month occupancy forecast; this module adds
a one-step ML cross-check, an honest ML-vs-Holt-Winters comparison, a leaderboard
and confidence intervals.

Target: `occupied_beds` (occupancy_pct is just occupied_beds / TOTAL_BEDS * 100),
so the ML output is directly comparable to the Holt-Winters occupied-beds series.

Leakage discipline (identical to src/revenue_ml.py):
  * Only features KNOWN BEFORE month t are used - lagged occupied beds / tenants /
    revenue / collection / electricity / arpu, plus the deterministic calendar.
  * Contemporaneous occupancy_pct / active_tenants / revenue / collection_rate /
    electricity_billed / arpu are month-t realisations -> EXCLUDED as leakage.
  * occupancy_pct_lag1 is excluded too: it equals occ_lag1 / TOTAL_BEDS, i.e. a
    perfect linear copy of occ_lag1 (no new information, singular for linear).
  * Chronological expanding-window (walk-forward) validation only.

No synthetic data - every value comes from the real `property_month` panel.

Run:  python -m src.occupancy_ml
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
from sklearn.ensemble import (GradientBoostingRegressor,  # noqa: E402
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import forecasting as tsf  # noqa: E402
from src import revenue_ml  # noqa: E402

TARGET = "occupied_beds"
# Leakage-safe features (all lagged / deterministic), built in _build_frame.
FEATURES = ["occ_lag1", "occ_lag2", "occ_lag3", "occ_lag12", "occ_roll3_lag",
            "active_tenants_lag1", "revenue_lag1", "revenue_lag2", "revenue_lag3",
            "collection_rate_lag1", "electricity_billed_lag1", "arpu_lag1",
            "month_num", "year"]
N_TEST = 12                    # evaluate the most recent 12 months (expanding)
HW_METHOD = "holt_winters"     # primary model's occupancy method
Z95 = 1.96                     # ~95% band multiplier on the walk-forward MAE


def _build_frame(pm: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe supervised frame from the real monthly occupancy panel."""
    df = pm.sort_values("billing_period").reset_index(drop=True).copy()
    occ = df[TARGET]
    df["occ_lag1"] = occ.shift(1)
    df["occ_lag2"] = occ.shift(2)
    df["occ_lag3"] = occ.shift(3)
    df["occ_lag12"] = occ.shift(12)
    df["occ_roll3_lag"] = occ.shift(1).rolling(3).mean()   # mean of t-1,t-2,t-3
    df["active_tenants_lag1"] = df["active_tenants"].shift(1)
    df["revenue_lag1"] = df["revenue"].shift(1)
    df["revenue_lag2"] = df["revenue"].shift(2)
    df["revenue_lag3"] = df["revenue"].shift(3)
    df["collection_rate_lag1"] = df["collection_rate"].shift(1)
    df["electricity_billed_lag1"] = df["electricity_billed"].shift(1)
    df["arpu_lag1"] = df["arpu"].shift(1)
    return df


def _models() -> dict:
    """Scaled linear + tree bank; optional boosters when installed."""
    m = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=config.RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            random_state=config.RANDOM_STATE, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, random_state=config.RANDOM_STATE,
            verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        m["CatBoost"] = CatBoostRegressor(
            iterations=300, learning_rate=0.05, random_state=config.RANDOM_STATE,
            verbose=0)
    except ImportError:
        pass
    return m


def run():
    t0 = time.time()
    pm = fe.build_all()["property_month"]
    df = _build_frame(pm)
    occ_series = df[TARGET].to_numpy(dtype=float)
    months = df["billing_period"].astype(str).to_numpy()

    # Usable rows have every lag feature (drops the first 12 months for occ_lag12).
    usable = df.dropna(subset=FEATURES).index.to_numpy()
    test_idx = usable[usable >= len(df) - N_TEST]
    if len(test_idx) < 4:
        test_idx = usable[-max(4, len(usable) // 3):]

    models = _models()
    preds = {name: [] for name in models}
    preds["HoltWinters"] = []
    actuals, test_months = [], []

    for i in test_idx:
        tr = [u for u in usable if u < i]          # strictly-past training rows
        if len(tr) < 6:
            continue
        X_tr = df.loc[tr, FEATURES].to_numpy(dtype=float)
        y_tr = df.loc[tr, TARGET].to_numpy(dtype=float)
        X_te = df.loc[[i], FEATURES].to_numpy(dtype=float)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds[name].append(float(model.predict(X_te)[0]))
        # Primary Holt-Winters at the SAME origin, one-step ahead (fair compare).
        # Uses the production _fit_forecast (holt_winters with its own fallback
        # on short seasonal windows) so this reflects deployed behaviour.
        hw = tsf._fit_forecast(occ_series[:i], 1, HW_METHOD)[0]
        preds["HoltWinters"].append(float(hw))
        actuals.append(float(occ_series[i]))
        test_months.append(months[i])

    actuals = np.array(actuals)

    rows = []
    for name, p in preds.items():
        m = utils.regression_metrics(actuals, np.array(p))
        m["Model"] = name
        rows.append(m)
    comp = (pd.DataFrame(rows).set_index("Model")[["MAE", "RMSE", "MAPE", "R2"]]
              .round(3).sort_values("MAPE"))

    # ML leaderboard (exclude Holt-Winters) + best ML model.
    ml_board = comp.drop(index="HoltWinters", errors="ignore")
    best_ml = ml_board.index[0]
    best_overall = comp.index[0]
    comp_out = comp.reset_index()
    comp_out["Best Model"] = comp_out["Model"] == best_overall
    comp_out.to_csv(config.OUT_DIR / "occupancy_model_comparison.csv", index=False)
    ml_board.to_csv(config.OUT_DIR / "occupancy_leaderboard.csv")

    # Walk-forward pred-vs-actual for the dashboard (best ML + Holt-Winters).
    pd.DataFrame({
        "billing_period": test_months, "actual": actuals,
        "ml_predicted": preds[best_ml], "hw_predicted": preds["HoltWinters"],
    }).to_csv(config.OUT_DIR / "occupancy_backtest_ml.csv", index=False)

    # Refit best ML on ALL usable rows; recursive 6-month forecast + 95% band.
    final = _models()[best_ml]
    Xall = df.loc[usable, FEATURES].to_numpy(dtype=float)
    yall = df.loc[usable, TARGET].to_numpy(dtype=float)
    final.fit(Xall, yall)

    best_mae = float(ml_board.loc[best_ml, "MAE"])
    band = round(Z95 * best_mae, 1)
    future = _future_forecast(df, final, band, steps=6)
    future.to_csv(config.OUT_DIR / "forecast_occupancy_ml.csv", index=False)

    import joblib
    joblib.dump(final, config.MODEL_DIR / "occupancy_ml_best.pkl")

    hw_mape = float(comp.loc["HoltWinters", "MAPE"])
    ml_mape = float(ml_board.loc[best_ml, "MAPE"])
    winner = "HoltWinters" if hw_mape <= ml_mape else best_ml
    nxt = future.iloc[0]
    hist_max = int(df[TARGET].max())
    meta = {
        "primary_model": "Holt-Winters (src/forecasting.py) - unchanged, "
                         "remains production baseline",
        "ml_model": best_ml,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "excluded_leakage_features": [
            "occupancy_pct (=occupied_beds/TOTAL_BEDS, contemporaneous)",
            "active_tenants / revenue / collection_rate / electricity_billed / "
            "arpu (month-t realisations)",
            "occupancy_pct_lag1 (perfect linear copy of occ_lag1)"],
        "n_train_months_total": int(len(usable)),
        "n_test_months": int(len(actuals)),
        "validation": "chronological expanding-window walk-forward (one-step)",
        "winner": winner,
        "hw_mape": round(hw_mape, 3),
        "ml_mape": round(ml_mape, 3),
        "hw_mae": round(float(comp.loc["HoltWinters", "MAE"]), 3),
        "ml_mae": round(best_mae, 3),
        "improvement_mape_pct_points": round(hw_mape - ml_mape, 3),
        "next_month_period": str(nxt["billing_period"]),
        "next_month_occupied_beds": int(nxt["occupied_beds"]),
        "next_month_occupancy_pct": float(nxt["occupancy_pct"]),
        "next_month_lower_beds": int(nxt["lower"]),
        "next_month_upper_beds": int(nxt["upper"]),
        "ci_method": f"+/- {Z95} x walk-forward MAE, clipped to [0, "
                     f"{config.TOTAL_BEDS}]",
        "total_beds_capacity": int(config.TOTAL_BEDS),
        "historical_max_occupied_beds": hist_max,
        "historical_max_occupancy_pct": round(hist_max / config.TOTAL_BEDS * 100, 2),
        "capacity_note": (
            f"Occupied beds never exceeded {hist_max} ("
            f"{hist_max / config.TOTAL_BEDS * 100:.1f}%) in {len(df)} months of "
            f"real data. A 192/100% point forecast is a capacity-clip artifact, "
            f"not a historically realistic value."),
        "trained_at": revenue_ml._utc_now(),
        "training_duration_sec": round(time.time() - t0, 1),
        "model_version": revenue_ml._utc_version(),
    }
    (config.OUT_DIR / "occupancy_model_metadata.json").write_text(
        json.dumps(meta, indent=2))

    _plot(test_months, actuals, preds, best_ml)

    print(f"Occupancy model comparison (walk-forward, {len(actuals)} test months):")
    print(comp.to_string())
    print(f"\nBest ML model: {best_ml} | Winner overall: {winner}")
    print(f"Next month ({nxt['billing_period']}): {int(nxt['occupied_beds'])} beds "
          f"({nxt['occupancy_pct']:.1f}%) | range {int(nxt['lower'])}-"
          f"{int(nxt['upper'])} beds")
    return comp


def _future_forecast(df, model, band: float, steps: int = 6) -> pd.DataFrame:
    """Recursive multi-step occupancy forecast + 95% band.

    Occupied-bed lags are fed recursively from the model's own predictions. The
    exogenous lagged drivers (tenants / revenue / collection / electricity / arpu)
    use recent-3-month persistence - the same non-synthetic pattern used by
    revenue_multivariate._future_forecast. Band = +/- Z95 * walk-forward MAE.
    """
    occ_hist = df[TARGET].tolist()
    recent = df.tail(3)
    persist = {c: float(recent[c].mean()) for c in
               ["active_tenants", "revenue", "collection_rate",
                "electricity_billed", "arpu"]}

    last_period = df["billing_period"].iloc[-1]
    if not isinstance(last_period, pd.Period):
        last_period = pd.Period(str(last_period), freq="M")
    periods = pd.period_range(last_period + 1, periods=steps, freq="M")

    rows = []
    for p in periods:
        feat = {
            "occ_lag1": occ_hist[-1], "occ_lag2": occ_hist[-2],
            "occ_lag3": occ_hist[-3],
            "occ_lag12": occ_hist[-12] if len(occ_hist) >= 12 else occ_hist[0],
            "occ_roll3_lag": float(np.mean(occ_hist[-3:])),
            "active_tenants_lag1": persist["active_tenants"],
            "revenue_lag1": persist["revenue"], "revenue_lag2": persist["revenue"],
            "revenue_lag3": persist["revenue"],
            "collection_rate_lag1": persist["collection_rate"],
            "electricity_billed_lag1": persist["electricity_billed"],
            "arpu_lag1": persist["arpu"],
            "month_num": p.month, "year": p.year,
        }
        vec = np.array([[feat[c] for c in FEATURES]], dtype=float)
        yhat = float(model.predict(vec)[0])
        occ_hist.append(yhat)
        cap = config.TOTAL_BEDS
        beds = int(round(min(max(yhat, 0), cap)))
        lo = int(round(min(max(yhat - band, 0), cap)))
        hi = int(round(min(max(yhat + band, 0), cap)))
        rows.append({"billing_period": str(p), "occupied_beds": beds,
                     "occupancy_pct": round(beds / cap * 100, 2),
                     "lower": lo, "upper": hi,
                     "lower_pct": round(lo / cap * 100, 2),
                     "upper_pct": round(hi / cap * 100, 2)})
    return pd.DataFrame(rows)


def _plot(months, actuals, preds, best_ml):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(months, actuals, marker="o", color="#2A9D8F", label="actual", lw=2)
    ax.plot(months, preds["HoltWinters"], marker="s", ls="--", color="#E76F51",
            label="Holt-Winters (primary)")
    ax.plot(months, preds[best_ml], marker="^", ls=":", color="#264653",
            label=f"ML ({best_ml})")
    ax.set_title("Occupied beds: actual vs Holt-Winters vs ML (walk-forward)")
    ax.set_ylabel("occupied beds")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "occupancy_model_comparison.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
