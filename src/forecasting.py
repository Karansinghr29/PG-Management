"""Time-series forecasting on the real 40-month property series.

Forecasts monthly revenue, active tenants and electricity cost from
`property_month` (built purely from invoices + electricity - no invented data).

Uses statsmodels Holt-Winters when installed; otherwise falls back to a
seasonal-naive + linear-trend estimator. Backtests the last `horizon` months to
report honest error (MAE / MAPE) before forecasting forward.

Run:  python -m src.forecasting
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import feature_engineering as fe  # noqa: E402

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


def _holt_winters(y, steps):
    model = ExponentialSmoothing(
        y, trend="add", seasonal="add", seasonal_periods=12,
        initialization_method="heuristic")
    return np.asarray(model.fit().forecast(steps), dtype=float)


def _linear_seasonal(y, steps):
    idx = np.arange(len(y))
    coef = np.polyfit(idx, y, 1)
    resid = y - np.polyval(coef, idx)
    season = np.array([resid[i::12].mean() if len(resid[i::12]) else 0
                       for i in range(12)])
    fut = np.arange(len(y), len(y) + steps)
    return np.polyval(coef, fut) + season[fut % 12]


def _seasonal_naive(y, steps):
    """Repeat the value from 12 months earlier (or last value if too short)."""
    if len(y) >= 12:
        return np.array([y[len(y) - 12 + (i % 12)] for i in range(steps)])
    return np.repeat(y[-1], steps)


def _methods():
    m = {"linear_seasonal": _linear_seasonal, "seasonal_naive": _seasonal_naive}
    if HAVE_SM:
        m = {"holt_winters": _holt_winters, **m}
    return m


def _fit_forecast(y: np.ndarray, steps: int, method: str | None = None) -> np.ndarray:
    """Return `steps`-ahead forecast. If `method` is None, use the best-known
    default (Holt-Winters when the series is long enough)."""
    y = np.asarray(y, dtype=float)
    methods = _methods()
    if method is None:
        method = "holt_winters" if (HAVE_SM and len(y) >= 24) else "linear_seasonal"
    try:
        return methods[method](y, steps)
    except Exception:
        return _linear_seasonal(y, steps)


def _mape(actual, pred):
    return float(np.nanmean(np.abs((actual - pred)
                / np.where(actual == 0, np.nan, actual))) * 100)


def _rolling_backtest(y: np.ndarray, method: str, horizon: int = 1,
                      n_windows: int = 6) -> dict:
    """Walk-forward (rolling-origin) validation: expand the training window one
    month at a time, forecast `horizon` ahead, average the error. No leakage -
    each forecast only sees data strictly before it."""
    y = np.asarray(y, dtype=float)
    min_train = max(24, len(y) - n_windows - horizon + 1)
    errs, preds, actuals, ends = [], [], [], []
    for end in range(min_train, len(y) - horizon + 1):
        train = y[:end]
        pred = _fit_forecast(train, horizon, method)[horizon - 1]
        actual = y[end + horizon - 1]
        errs.append(abs(actual - pred)); preds.append(pred)
        actuals.append(actual); ends.append(end + horizon - 1)
    if not errs:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"),
                "windows": 0, "preds": [], "actuals": [], "positions": []}
    a, p = np.array(actuals), np.array(preds)
    return {"MAE": float(np.mean(errs)),
            "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
            "MAPE": _mape(a, p),
            "windows": len(errs),
            "preds": preds, "actuals": actuals, "positions": ends}


def select_method(y: np.ndarray) -> tuple[str, dict]:
    """Pick the forecasting method with the lowest walk-forward MAPE."""
    best, best_bt = None, {"MAPE": float("inf")}
    for name in _methods():
        bt = _rolling_backtest(y, name)
        if bt["MAPE"] == bt["MAPE"] and bt["MAPE"] < best_bt["MAPE"]:  # not NaN
            best, best_bt = name, bt
    if best is None:                       # all NaN (series too short)
        best, best_bt = "linear_seasonal", _rolling_backtest(y, "linear_seasonal")
    return best, best_bt


def _clean_series(pm: pd.DataFrame, col: str) -> np.ndarray:
    """Interpolate gaps so months without electricity data don't break the fit."""
    return (pm[col].astype(float).interpolate().bfill().ffill().to_numpy())


def forecast_series(pm: pd.DataFrame, col: str, steps: int = 6,
                    method: str | None = None) -> pd.DataFrame:
    y = _clean_series(pm, col)
    if method is None:
        method, _ = select_method(y)
    fc = _fit_forecast(y, steps, method)
    last = pm["billing_period"].max()
    future = pd.period_range(last + 1, periods=steps, freq="M")
    return pd.DataFrame({"billing_period": future.astype(str), col: fc})


# --------------------------------------------------------------------------- #
# Feature-based revenue model + Holt-Winters vs ML comparison (auto-select)
# --------------------------------------------------------------------------- #
def _metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return {"MAE": float(np.mean(np.abs(a - p))),
            "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
            "MAPE": _mape(a, p)}


def _feature_model():
    """XGBoost if installed, else a gradient-boosting fallback. Not hardcoded as
    the forecast winner — it competes with Holt-Winters on the same windows."""
    try:
        from xgboost import XGBRegressor
        return "XGBoost", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            random_state=config.RANDOM_STATE, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return "GradientBoosting", GradientBoostingRegressor(
            random_state=config.RANDOM_STATE)


# Leakage-safe supervised features (all lagged / calendar) — uses the new
# occupancy lag features incl. occupancy_pct_lag3.
REV_FEATURES = ["rev_lag1", "rev_lag2", "rev_lag3", "rev_lag12", "rev_roll3_lag",
                "tenants_lag1", "occupancy_pct_lag1", "occupancy_pct_lag3",
                "month_num", "year"]


def _revenue_frame(pm: pd.DataFrame) -> pd.DataFrame:
    df = pm.sort_values("billing_period").reset_index(drop=True).copy()
    if "occupancy_pct_lag3" not in df.columns:
        df["occupancy_pct_lag3"] = df["occupancy_pct"].shift(3)
    rev = df["revenue"]
    df["rev_lag1"] = rev.shift(1)
    df["rev_lag2"] = rev.shift(2)
    df["rev_lag3"] = rev.shift(3)
    df["rev_lag12"] = rev.shift(12)
    df["rev_roll3_lag"] = rev.shift(1).rolling(3).mean()
    df["tenants_lag1"] = df["active_tenants"].shift(1)
    return df


def revenue_comparison(pm: pd.DataFrame, n_test: int = 12):
    """Walk-forward one-step: Holt-Winters vs feature-based ML on identical test
    months. Returns (comparison_df, winner, ml_name, months, actual, hw, ml, n)."""
    df = _revenue_frame(pm)
    feats = [f for f in REV_FEATURES if f in df.columns]
    rev = df["revenue"].to_numpy(float)
    months = df["billing_period"].astype(str).to_numpy()
    usable = df.dropna(subset=feats).index.to_numpy()
    test_idx = [i for i in usable[usable >= len(df) - n_test]
                if len([u for u in usable if u < i]) >= 6]
    ml_name, model = _feature_model()
    hw_p, ml_p, actual, tmonths = [], [], [], []
    for i in test_idx:
        tr = [u for u in usable if u < i]
        X_tr = df.loc[tr, feats].to_numpy(float)
        y_tr = df.loc[tr, "revenue"].to_numpy(float)
        model.fit(X_tr, y_tr)
        ml_p.append(float(model.predict(df.loc[[i], feats].to_numpy(float))[0]))
        hw_p.append(float(_fit_forecast(rev[:i], 1, "holt_winters")[0]))
        actual.append(float(rev[i])); tmonths.append(months[i])
    actual = np.array(actual)
    comp = pd.DataFrame([
        {"Model": "Holt-Winters", **_metrics(actual, hw_p)},
        {"Model": ml_name, **_metrics(actual, ml_p)},
    ])[["Model", "MAE", "RMSE", "MAPE"]].round(2)
    winner = comp.sort_values("MAPE").iloc[0]["Model"]      # lowest MAPE, not hardcoded
    return comp, winner, ml_name, tmonths, actual, hw_p, ml_p, len(actual)


def forecast_live(pm: pd.DataFrame, steps: int = 6,
                  series: tuple[str, ...] = ("revenue", "occupied_beds")) -> dict:
    """In-memory forecast for live consumers (e.g. AI Recommendations).

    Reuses the exact same model selection / forecasting logic as ``run`` but
    performs NO file writes and NO plotting, and does not rebuild features — the
    caller passes the already-built ``property_month``. Returns the same shapes
    the AI page previously read from disk:
        {"forecast_summary": DataFrame(series, method, MAE, MAPE, windows,
                                       next_month),
         "occupancy_forecast": DataFrame(billing_period, occupancy_pct) | None}
    The model is unchanged; only the delivery (live vs persisted CSV) differs.
    """
    pm = pm.copy()
    if "occupancy_pct_lag3" not in pm.columns:
        pm["occupancy_pct_lag3"] = pm["occupancy_pct"].shift(3)
    rows: list[dict] = []
    occ_df = None
    for col in series:
        y = _clean_series(pm, col)
        method, bt = select_method(y)          # same walk-forward selection
        fdf = forecast_series(pm, col, steps, method)
        if col == "occupied_beds":
            fdf[col] = fdf[col].clip(0, config.TOTAL_BEDS).round().astype(int)
            fdf["occupancy_pct"] = (fdf[col] / config.TOTAL_BEDS * 100).round(2)
            occ_df = fdf[["billing_period", "occupancy_pct"]].copy()
        rows.append({"series": col, "method": method,
                     "MAE": round(bt["MAE"], 1),
                     "RMSE": round(bt.get("RMSE", float("nan")), 1),
                     "MAPE": round(bt["MAPE"], 2),
                     "windows": bt["windows"],
                     "next_month": round(float(fdf[col].iloc[0]), 1)})
    return {"forecast_summary": pd.DataFrame(rows), "occupancy_forecast": occ_df}


def run(steps: int = 6):
    pm = fe.build_all()["property_month"].copy()
    # Requirement: extend the feature set with occupancy_pct_lag3 if missing.
    # (Only added when absent; existing lag columns are untouched.)
    if "occupancy_pct_lag3" not in pm.columns:
        pm["occupancy_pct_lag3"] = pm["occupancy_pct"].shift(3)
    engine = "Holt-Winters (statsmodels)" if HAVE_SM else "linear+seasonal fallback"
    print(f"Forecast engine: {engine}\n")

    targets = ["revenue", "active_tenants", "elec_cost", "occupied_beds"]
    summary = []
    fig, axes = plt.subplots(len(targets), 1, figsize=(13, 11))
    for ax, col in zip(axes, targets):
        y = _clean_series(pm, col)
        method, bt = select_method(y)          # walk-forward model selection
        fdf = forecast_series(pm, col, steps, method)
        if col == "occupied_beds":             # occupancy is bounded [0, 100]
            fdf[col] = (fdf[col].clip(0, config.TOTAL_BEDS).round().astype(int))
            fdf ["occupancy_pct"] = ( fdf[col] / config.TOTAL_BEDS * 100).round(2)
            fdf.to_csv(config.OUT_DIR / "forecast_occupancy_pct.csv", index=False)   
        summary.append({"series": col, "method": method,
                        "MAE": round(bt["MAE"], 1),
                        "RMSE": round(bt.get("RMSE", float("nan")), 1),
                        "MAPE": round(bt["MAPE"], 2),
                        "windows": bt["windows"],
                        "next_month": round(float(fdf[col].iloc[0]), 1)})
        # Persist walk-forward pred-vs-actual for the dashboard accuracy chart.
        if bt["windows"]:
            periods = pm["billing_period"].astype(str).to_numpy()
            pd.DataFrame({
                "billing_period": [periods[p] for p in bt["positions"]],
                "actual": bt["actuals"], "predicted": bt["preds"],
            }).to_csv(config.OUT_DIR / f"backtest_{col}.csv", index=False)
        hist_x = pm["billing_period"].astype(str)
        ax.plot(hist_x, y, marker="o", label="actual")
        ax.plot(fdf["billing_period"], fdf[col], marker="s", ls="--",
                color="#C44536", label="forecast")
        ax.set_title(f"{col} - {steps}-mo forecast  [{method}]  "
                     f"walk-forward MAPE {bt['MAPE']:.1f}% (n={bt['windows']})")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.legend()
        fdf.to_csv(config.OUT_DIR / f"forecast_{col}.csv", index=False)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "forecast.png", dpi=120)
    plt.close(fig)

    s = pd.DataFrame(summary)
    s.to_csv(config.OUT_DIR / "forecast_summary.csv", index=False)

    # ---- Revenue: Holt-Winters vs feature-based ML (auto-select winner) ----- #
    import json
    comp, winner, ml_name, _tm, _act, _hw, _ml, n = revenue_comparison(pm)
    comp.to_csv(config.OUT_DIR / "revenue_forecast_comparison.csv", index=False)
    (config.OUT_DIR / "revenue_forecast_selected.json").write_text(json.dumps({
        "candidates": ["Holt-Winters", ml_name],
        "winner": winner, "n_test_months": int(n),
        "metrics": comp.set_index("Model").round(2).to_dict("index"),
    }, indent=2))

    print(">> Time-series forecast summary")
    print(s.round(2).to_string(index=False))
    print(f"\n>> Revenue model comparison (walk-forward, {n} test months)")
    print(comp.to_string(index=False))
    print(f"Selected revenue forecasting model: {winner}")
    occ = s[s["series"] == "occupied_beds"].iloc[0]
    print(f"\n>> Occupancy forecast: method={occ['method']} "
          f"MAPE={occ['MAPE']}% next-month occupied={int(occ['next_month'])} "
          f"({occ['next_month']/config.TOTAL_BEDS*100:.1f}%)")
    print(f"\nfigures -> {config.FIG_DIR / 'forecast.png'}")
    return s


if __name__ == "__main__":
    run()