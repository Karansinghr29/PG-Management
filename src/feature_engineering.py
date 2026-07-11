"""Step 5 - feature engineering.

Builds analysis/model-ready feature tables from the cleaned data:
  * invoice_features   - per-invoice, for late-payment & revenue models
  * electricity_features - per apartment-month
  * property_month     - portfolio KPIs per month
  * bed_features       - current occupancy view
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from src import preprocessing  # noqa: E402


def invoice_features(inv: pd.DataFrame) -> pd.DataFrame:
    df = inv.copy()
    df["month_num"] = df["billing_period"].dt.month
    df["year"] = df["billing_period"].dt.year
    df["quarter"] = df["billing_period"].dt.quarter
    df["elec_share"] = np.where(df["total_amount"] > 0,
                                df["electricity_amount"] / df["total_amount"], 0)
    df["has_other_charges"] = (df["other_charges"] > 0).astype(int)
    # Historic behaviour of the tenant up to (but not including) this invoice.
    df["prior_unpaid_ratio"] = np.where(
        df["prior_invoices"] > 0, df["prior_unpaid"] / df["prior_invoices"], 0)
    df["is_new_tenant"] = (df["prior_invoices"] == 0).astype(int)
    df = df.sort_values(["tenant_id", "billing_period"])
    df["rent_growth"] = (
        df.groupby("tenant_id")["rent_amount"].pct_change().fillna(0))
    return df


def electricity_features(elec: pd.DataFrame) -> pd.DataFrame:
    df = elec.copy()
    df["month_num"] = df["billing_period"].dt.month
    df["year"] = df["billing_period"].dt.year
    df["abs_deviation"] = df["deviation_from_avg"].abs()
    df["high_usage_flag"] = (df["deviation_from_avg"] > df["apt_avg_units"]).astype(int)
    df = df.sort_values(["apartment_code", "billing_period"])
    df["units_mom"] = (
        df.groupby("apartment_code")["units_consumed"].pct_change().fillna(0))
    return df


def occupancy_month(bookings: pd.DataFrame,
                    period_index: pd.PeriodIndex) -> pd.DataFrame:
    """Real historical monthly occupancy from stay/booking history.

    A bed is occupied in month M if the tenant onboarded on/before M-end and had
    not yet exited by M-start. Denominator = physical bed capacity (config).
    No synthetic data - every count comes from real onboarding/exit/notice dates.
    """
    b = bookings.copy()
    # Booking dates are parsed tz-aware (UTC); match the month bounds to them.
    tz = getattr(b["onboarding_date"].dtype, "tz", None)
    rows = []
    for m in period_index:
        s, e = m.start_time, m.end_time
        if tz is not None:
            s, e = s.tz_localize(tz), e.tz_localize(tz)
        live = b[(b["onboarding_date"] <= e)
                 & (b["actual_exit_date"].isna() | (b["actual_exit_date"] >= s))]
        rows.append({
            "billing_period": m,
            "occupied_beds": int(live["bed_id"].nunique()),
            "active_tenants_occ": int(live["tenant_id"].nunique()),
            "move_ins": int(((b["onboarding_date"] >= s)
                             & (b["onboarding_date"] <= e)).sum()),
            "move_outs": int(((b["actual_exit_date"] >= s)
                              & (b["actual_exit_date"] <= e)).sum()),
            "notice_count": int(((b["notice_date"] >= s)
                                 & (b["notice_date"] <= e)).sum()),
            "new_bookings": int(((b["booking_date"] >= s)
                                 & (b["booking_date"] <= e)).sum()),
            "avg_monthly_rental": float(live["monthly_rental"].mean()),
        })
    df = pd.DataFrame(rows)
    df["occupancy_pct"] = (df["occupied_beds"] / config.TOTAL_BEDS * 100).round(2)
    df["vacant_beds"] = config.TOTAL_BEDS - df["occupied_beds"]
    return df


def property_month(inv: pd.DataFrame, elec: pd.DataFrame,
                   bookings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Portfolio-level monthly KPI table (revenue, collection, occupancy proxy)."""
    rev = (inv.groupby("billing_period")
              .agg(revenue=("total_amount", "sum"),
                   rent=("rent_amount", "sum"),
                   electricity_billed=("electricity_amount", "sum"),
                   invoices=("invoice_id", "count"),
                   active_tenants=("tenant_id", "nunique"),
                   unpaid=("is_unpaid", "sum"))
              .reset_index())
    rev["collection_rate"] = 1 - rev["unpaid"] / rev["invoices"]
    rev["revenue_at_risk"] = (
        inv.assign(risk=inv["total_amount"] * inv["is_unpaid"])
           .groupby("billing_period")["risk"].sum().values)
    rev["arpu"] = rev["revenue"] / rev["active_tenants"]        # avg rev per tenant
    e = (elec.groupby("billing_period")
             .agg(units=("units_consumed", "sum"),
                  elec_cost=("amount", "sum")).reset_index())
    out = rev.merge(e, on="billing_period", how="left")
    # Join real monthly occupancy from booking history (manager requirement).
    if bookings is not None:
        occ = occupancy_month(bookings, list(out["billing_period"]))
        out = out.merge(occ, on="billing_period", how="left")
    out["month_num"] = out["billing_period"].dt.month
    out["year"] = out["billing_period"].dt.year
    out = out.sort_values("billing_period").reset_index(drop=True)
    # Lag / rolling features for forecasting (real monthly series).
    for lag in (1, 2, 3, 12):
        out[f"revenue_lag{lag}"] = out["revenue"].shift(lag)
    out["revenue_roll3"] = out["revenue"].rolling(3).mean()
    out["revenue_mom"] = out["revenue"].pct_change()
    out["tenants_lag1"] = out["active_tenants"].shift(1)
    # Lagged occupancy features (known before month t -> leakage-safe drivers).
    for col in ("occupancy_pct", "occupied_beds", "active_tenants_occ",
                "move_ins", "move_outs", "notice_count", "new_bookings",
                "avg_monthly_rental"):
        if col in out.columns:
            out[f"{col}_lag1"] = out[col].shift(1)
    return out


def tenant_features(inv: pd.DataFrame) -> pd.DataFrame:
    """Per-tenant billing profile. Built ONLY by grouping invoices on the real
    `tenant_id` key (same table) - no cross-table joins, no invented mapping."""
    g = inv.sort_values("billing_period").groupby("tenant_id")
    out = g.agg(
        invoices=("invoice_id", "count"),
        first_month=("billing_period", "min"),
        last_month=("billing_period", "max"),
        total_billed=("total_amount", "sum"),
        total_rent=("rent_amount", "sum"),
        total_electricity=("electricity_amount", "sum"),
        unpaid_count=("is_unpaid", "sum"),
        avg_rent=("rent_amount", "mean"),
    ).reset_index()
    # Tenure in months from real billing span (inclusive).
    out["tenure_months"] = (
        (out["last_month"].dt.year - out["first_month"].dt.year) * 12
        + (out["last_month"].dt.month - out["first_month"].dt.month) + 1)
    out["unpaid_ratio"] = out["unpaid_count"] / out["invoices"]
    out["avg_monthly_billing"] = out["total_billed"] / out["invoices"]
    # Billing-based lifetime value proxy (paid billings only).
    paid = (inv.assign(paid_amt=inv["total_amount"] * (1 - inv["is_unpaid"]))
               .groupby("tenant_id")["paid_amt"].sum())
    out["ltv_paid"] = out["tenant_id"].map(paid)
    return out


def bed_features(beds: pd.DataFrame) -> pd.DataFrame:
    df = beds.copy()
    df["is_occupied"] = (df["bed_lifecycle_status"] == "occupied").astype(int)
    df["on_notice"] = (df["bed_lifecycle_status"] == "notice").astype(int)
    df["is_vacant"] = (df["bed_lifecycle_status"] == "vacant").astype(int)
    return df


def build_all(cleaned: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    cleaned = cleaned or preprocessing.clean_all()
    return {
        "invoice_features": invoice_features(cleaned["invoices"]),
        "electricity_features": electricity_features(cleaned["electricity"]),
        "property_month": property_month(cleaned["invoices"], cleaned["electricity"],
                                         cleaned.get("bookings")),
        "tenant_features": tenant_features(cleaned["invoices"]),
        "bed_features": bed_features(cleaned["beds_snapshot"]),
    }


if __name__ == "__main__":
    feats = build_all()
    for name, df in feats.items():
        out = config.OUT_DIR / f"feat_{name}.csv"
        df.to_csv(out, index=False)
        print(f"{name:22} {df.shape} -> {out.name}")
