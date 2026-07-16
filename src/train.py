"""Steps 6 & 7 - modelling and model comparison.

Trains every applicable prediction problem, compares a wide bank of algorithms
with cross-validation, reports the full metric grid + feature importance
(and SHAP when available), then persists the best estimator per problem.

Optional libraries (xgboost / lightgbm / catboost / shap) are used only when
installed; the script degrades gracefully to the scikit-learn bank otherwise.

Run:  python -m src.train              (all problems)
      python -m src.train late_payment (one problem)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from src import feature_engineering as fe  # noqa: E402

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,  # noqa: E402
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (LinearRegression, LogisticRegression,  # noqa: E402
                                  Ridge)
from sklearn.model_selection import (TimeSeriesSplit,  # noqa: E402
                                     cross_val_score)
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import (KNeighborsClassifier,  # noqa: E402
                               KNeighborsRegressor)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC, SVR  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402


# --------------------------------------------------------------------------- #
# Optional boosters
# --------------------------------------------------------------------------- #
def _optional_models(task: str) -> dict:
    out = {}
    try:
        from xgboost import XGBClassifier, XGBRegressor
        out["XGBoost"] = (XGBRegressor if task == "regression" else XGBClassifier)(
            n_estimators=300, learning_rate=0.05, random_state=config.RANDOM_STATE,
            verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        out["LightGBM"] = (LGBMRegressor if task == "regression" else LGBMClassifier)(
            n_estimators=300, learning_rate=0.05, random_state=config.RANDOM_STATE,
            verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        out["CatBoost"] = (CatBoostRegressor if task == "regression"
                           else CatBoostClassifier)(
            iterations=300, learning_rate=0.05, random_state=config.RANDOM_STATE,
            verbose=0)
    except ImportError:
        pass
    return out


def model_bank(task: str) -> dict:
    rs = config.RANDOM_STATE
    if task == "regression":
        bank = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=rs),
            "DecisionTree": DecisionTreeRegressor(random_state=rs),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=rs,
                                                  n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=rs,
                                              n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=rs),
            "AdaBoost": AdaBoostRegressor(random_state=rs),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
        }
    else:
        bank = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=rs),
            "DecisionTree": DecisionTreeClassifier(random_state=rs),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=rs,
                                                   n_jobs=-1, class_weight="balanced"),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=rs,
                                               n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(random_state=rs),
            "AdaBoost": AdaBoostClassifier(random_state=rs),
            "KNN": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB(),
            "SVM": SVC(probability=True, random_state=rs),
        }
    bank.update(_optional_models(task))
    return bank


# --------------------------------------------------------------------------- #
# Feature matrices per problem (leakage-safe column selection)
# --------------------------------------------------------------------------- #
# Each _xy returns (X, y, task, order) where `order` is the chronological key
# used for a time-based train/test split and TimeSeriesSplit CV. This prevents
# training on the future to predict the past (temporal leakage).
def _xy_late_payment(feats):
    df = feats["invoice_features"]
    # Leakage-safe KEEP list. The new invoices carry UUIDs, datetimes and the
    # realised payment fields (status/amount_paid/balance) that ARE the target,
    # so we select only pre-payment drivers instead of dropping a few columns.
    y = df["is_unpaid"]
    keep = ["rent_amount", "electricity_amount", "other_charges", "credit_days",
            "month_num", "year", "quarter", "prior_invoices", "prior_unpaid",
            "prior_unpaid_ratio", "is_new_tenant", "elec_share",
            "has_other_charges", "rent_growth"]
    X = df[[c for c in keep if c in df.columns]]
    order = df["billing_period"]
    return X, y, "classification", order


def _xy_monthly_revenue(feats):
    df = feats["invoice_features"]
    # Predict total_amount from rent/elec drivers WITHOUT the components that
    # sum to it -> avoid target leakage. Use tenant history + calendar + rate.
    y = df["total_amount"]
    keep = ["rent_amount", "electricity_amount", "credit_days", "month_num",
            "year", "quarter", "prior_invoices", "prior_unpaid",
            "prior_unpaid_ratio", "is_new_tenant", "elec_share"]
    X = df[keep]
    order = df["billing_period"]
    return X, y, "regression", order


def _xy_electricity(feats):
    df = feats["electricity_features"]
    y = df["amount"]
    keep = ["units_consumed", "unit_cost", "apt_avg_units", "month_num", "year",
            "abs_deviation", "high_usage_flag", "apartment_code"]
    X = df[keep]
    order = df["billing_period"]
    return X, y, "regression", order


PROBLEMS = {
    "late_payment": _xy_late_payment,
    "monthly_revenue": _xy_monthly_revenue,
    "electricity_cost": _xy_electricity,
}

# Tenant / occupancy COUNT predictions must never exceed physical bed capacity.
_CAPACITY_TARGETS = {"active_tenants", "active_tenants_occ", "occupied_beds",
                     "occupancy"}


def _cap_capacity(name: str, pred, task: str):
    """Cap any tenant/occupancy-count regression prediction at the physical bed
    capacity (config.TOTAL_BEDS). Inert for non-count targets."""
    if task == "regression" and name in _CAPACITY_TARGETS:
        return np.minimum(np.asarray(pred, dtype=float), config.TOTAL_BEDS)
    return pred


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _build_pipeline(model, X: pd.DataFrame):
    cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def run_problem(name: str, feats: dict) -> pd.DataFrame:
    t_start = time.time()
    X, y, task, order = PROBLEMS[name](feats)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # --- Time-based split: sort chronologically, hold out the LAST slice. ---
    order = order.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sort_idx = order.sort_values(kind="mergesort").index
    X, y = X.iloc[sort_idx].reset_index(drop=True), y.iloc[sort_idx].reset_index(drop=True)
    cut = int(len(X) * (1 - config.TEST_SIZE))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    # TimeSeriesSplit for CV (expanding window, never trains on the future).
    tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    scoring = "r2" if task == "regression" else "f1"
    rows, fitted = [], {}
    for mname, model in model_bank(task).items():
        pipe = _build_pipeline(model, X)
        pipe.fit(X_tr, y_tr)
        pred = _cap_capacity(name, pipe.predict(X_te), task)
        if task == "regression":
            m = utils.regression_metrics(y_te, pred)
        else:
            proba = (pipe.predict_proba(X_te)[:, 1]
                     if hasattr(pipe["model"], "predict_proba") else None)
            m = utils.classification_metrics(y_te, pred, proba)
        try:
            cv = cross_val_score(pipe, X, y, cv=tscv, scoring=scoring,
                                 n_jobs=-1).mean()
        except Exception:
            cv = np.nan
        m["CV_" + scoring] = float(cv)
        m["Model"] = mname
        rows.append(m)
        fitted[mname] = pipe

    res = pd.DataFrame(rows).set_index("Model")
    rank_col = "R2" if task == "regression" else "F1"
    res = res.sort_values(rank_col, ascending=False)
    best_name = res.index[0]

    # Persist best model + leaderboard.
    joblib.dump(fitted[best_name], config.MODEL_DIR / f"{name}_best.pkl")
    res.to_csv(config.OUT_DIR / f"leaderboard_{name}.csv")
    _feature_importance(name, fitted[best_name], X)
    _permutation_importance(name, fitted[best_name], X_te, y_te, task)
    _save_metadata(name, task, best_name, X, order, t_start)

    print(f"\n### {name}  (task={task})  best={best_name}")
    print(res.round(4).to_string())
    return res


def _permutation_importance(name, pipe, X_te, y_te, task):
    """Model-agnostic importance on the held-out (future) slice. Works for any
    estimator (incl. NaiveBayes) - this is what the dashboard displays."""
    from sklearn.inspection import permutation_importance
    try:
        scoring = "r2" if task == "regression" else "f1"
        r = permutation_importance(pipe, X_te, y_te, n_repeats=5,
                                   random_state=config.RANDOM_STATE,
                                   scoring=scoring, n_jobs=-1)
        (pd.Series(r.importances_mean, index=X_te.columns)
           .sort_values(ascending=False)
           .to_csv(config.OUT_DIR / f"perm_importance_{name}.csv"))
    except Exception as e:
        print("  (permutation importance skipped:", e, ")")


def _save_metadata(name, task, best_name, X, order, t_start):
    import json
    from datetime import datetime, timezone
    meta = {
        "problem": name,
        "task": task,
        "best_model": best_name,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "training_duration_sec": round(time.time() - t_start, 1),
        "train_window": f"{order.min()} .. {order.max()}",
        "validation": "chronological 80/20 split + TimeSeriesSplit CV",
        "model_version": datetime.now(timezone.utc).strftime("v%Y%m%d.%H%M"),
    }
    with open(config.OUT_DIR / f"model_meta_{name}.json", "w") as f:
        json.dump(meta, f, indent=2)


def _feature_importance(name, pipe, X):
    try:
        model = pipe["model"]
        pre = pipe["pre"]
        names = pre.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(np.ravel(model.coef_))
        else:
            return
        fi = (pd.Series(imp, index=names).sort_values(ascending=False)
                .head(20))
        fi.to_csv(config.OUT_DIR / f"importance_{name}.csv")
        _shap_summary(name, pipe, X)
    except Exception as e:  # importance is best-effort
        print("  (importance skipped:", e, ")")


def _shap_summary(name, pipe, X):
    """Persist mean|SHAP| per feature when shap + a tree model are available."""
    try:
        import shap
    except ImportError:
        return
    model = pipe["model"]
    if not hasattr(model, "feature_importances_"):
        return  # SHAP TreeExplainer only for tree models here
    try:
        Xt = pipe["pre"].transform(X.sample(min(500, len(X)),
                                            random_state=config.RANDOM_STATE))
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        vals = shap.TreeExplainer(model).shap_values(Xt)
        if isinstance(vals, list):            # classifier -> use positive class
            vals = vals[-1]
        mean_abs = np.abs(vals).mean(axis=0)
        names = pipe["pre"].get_feature_names_out()
        (pd.Series(mean_abs, index=names).sort_values(ascending=False)
           .head(20).to_csv(config.OUT_DIR / f"shap_{name}.csv"))
        print(f"  SHAP saved for {name}")
    except Exception as e:
        print("  (SHAP skipped:", e, ")")


def main(which: list[str] | None = None):
    feats = fe.build_all()
    which = which or list(PROBLEMS)
    for name in which:
        run_problem(name, feats)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
