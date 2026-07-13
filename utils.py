"""Shared helpers: loaders, cleaning primitives and metric wrappers."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
from pathlib import Path
def load_raw(name: str) -> pd.DataFrame:
    path = config.RAW_DIR / config.RAW_FILES[name]
    return pd.read_csv(path)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load every raw table into a dict keyed by business name."""
    return {name: load_raw(name) for name in config.RAW_FILES}


# --------------------------------------------------------------------------- #
# Cleaning primitives
# --------------------------------------------------------------------------- #
def normalise_case(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Lower-case + strip categorical text so 'Male'/'male' collapse."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
    return df


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def drop_dead_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])


def billing_month_to_period(series: pd.Series) -> pd.Series:
    """Convert 'Apr-23' or '2023-04' style strings to a monthly Period."""
    s = series.astype("string")
    # Try the 'YYYY-MM' form first, then the 'Mon-YY' form.
    out = pd.to_datetime(s, format="%Y-%m", errors="coerce")
    mask = out.isna()
    out[mask] = pd.to_datetime(s[mask], format="%b-%y", errors="coerce")
    return out.dt.to_period("M")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": float(r2_score(y_true, y_pred))}


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["ROC_AUC"] = float("nan")
    return out
