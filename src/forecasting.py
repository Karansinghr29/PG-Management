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
        return {"MAE": float("nan"), "MAPE": float("nan"), "windows": 0,
                "preds": [], "actuals": [], "positions": []}
    return {"MAE": float(np.mean(errs)),
            "MAPE": _mape(np.array(actuals), np.array(preds)),
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


def run(steps: int = 6):
    pm = fe.build_all()["property_month"]
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
                        "MAE": round(bt["MAE"], 1), "MAPE": round(bt["MAPE"], 2),
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
    print(s.round(2).to_string(index=False))
    print(f"\nfigures -> {config.FIG_DIR / 'forecast.png'}")
    return s


if __name__ == "__main__":
    run()