"""Step 5 - feature engineering (Phase-2 migration: new production datasets).

Builds analysis/model-ready feature tables from the cleaned new data. Column
names/contracts the rest of the pipeline expects are preserved; missing inputs
are DERIVED from the new sources (no synthetic data):
  * invoice_features    - per-invoice; credit_days = due-invoice, prior_* rebuilt
  * electricity_features - per apartment (apt averages/deviation derived)
  * property_month      - portfolio monthly KPIs + new business metrics
  * tenant_features     - per-tenant billing profile
  * bed_features        - current bed view (derived snapshot)
  * apartment_features  - NEW: per-apartment occupancy/rent/electricity/notices

New monthly metrics added to property_month: collections, outstanding_balance,
monthly_expenses, net_revenue, average_rent, revenue_per_occupied_bed,
collection_rate_amount, monthly_cash_collected. Collections use the payments
ledger joined by tenant_allotment_id (primary) / tenant_id (fallback).
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


def _block(code: pd.Series) -> pd.Series:
    return code.astype("string").str.extract(r"^([A-Za-z]+)")[0]


# --------------------------------------------------------------------------- #
# Per-invoice features
# --------------------------------------------------------------------------- #
def invoice_features(inv: pd.DataFrame) -> pd.DataFrame:
    df = inv.copy()
    df["month_num"] = df["billing_period"].dt.month
    df["year"] = df["billing_period"].dt.year
    df["quarter"] = df["billing_period"].dt.quarter
    # credit_days no longer in the raw data -> derive from due_date - invoice_date.
    if "credit_days" not in df.columns and {"due_date", "invoice_date"} <= set(df.columns):
        df["credit_days"] = (df["due_date"] - df["invoice_date"]).dt.days
    df["credit_days"] = df.get("credit_days", 0)
    df["elec_share"] = np.where(df["total_amount"] > 0,
                                df["electricity_amount"] / df["total_amount"], 0)
    df["has_other_charges"] = (df["other_charges"] > 0).astype(int)
    # prior_invoices / prior_unpaid rebuilt as real tenant history up to (but not
    # including) this invoice (the new invoices have no prior_* columns).
    df = df.sort_values(["tenant_id", "billing_period"])
    grp = df.groupby("tenant_id")
    df["prior_invoices"] = grp.cumcount()
    df["prior_unpaid"] = grp["is_unpaid"].cumsum() - df["is_unpaid"]
    df["prior_unpaid_ratio"] = np.where(
        df["prior_invoices"] > 0, df["prior_unpaid"] / df["prior_invoices"], 0)
    df["is_new_tenant"] = (df["prior_invoices"] == 0).astype(int)
    df["rent_growth"] = grp["rent_amount"].pct_change().fillna(0)
    return df


# --------------------------------------------------------------------------- #
# Per-apartment electricity features
# --------------------------------------------------------------------------- #
def electricity_features(elec: pd.DataFrame) -> pd.DataFrame:
    df = elec.copy()
    df["month_num"] = df["billing_period"].dt.month
    df["year"] = df["billing_period"].dt.year
    # apt averages / deviation are no longer supplied -> derive from readings.
    apt_avg = df.groupby("apartment_code")["units_consumed"].transform("mean")
    df["apt_avg_units"] = apt_avg
    df["deviation_from_avg"] = df["units_consumed"] - apt_avg
    df["abs_deviation"] = df["deviation_from_avg"].abs()
    df["high_usage_flag"] = (df["deviation_from_avg"] > df["apt_avg_units"]).astype(int)
    df["unit_cost"] = np.where(df["units_consumed"] > 0,
                               df["amount"] / df["units_consumed"], 0)
    df = df.sort_values(["apartment_code", "billing_period"])
    df["units_mom"] = (
        df.groupby("apartment_code")["units_consumed"].pct_change().fillna(0))
    return df


# --------------------------------------------------------------------------- #
# Real monthly occupancy from booking history
# --------------------------------------------------------------------------- #
def occupancy_month(bookings: pd.DataFrame, period_index) -> pd.DataFrame:
    """Real monthly occupancy from stay history. A bed is occupied in month M if
    onboarded on/before M-end and not yet exited by M-start. New bookings are
    keyed by bed_code/full_name (no UUIDs) and have no booking_date, so
    new_bookings is proxied by onboardings in the month. No synthetic data."""
    b = bookings.copy()
    # bed_code is only unique WITHIN an apartment (reused across apartments), so a
    # physical bed = apartment_code + bed_code.
    b["_bed"] = (b["apartment_code"].astype("string") + "|"
                 + b["bed_code"].astype("string"))
    tz = getattr(b["onboarding_date"].dtype, "tz", None)
    rows = []
    for m in period_index:
        s, e = m.start_time, m.end_time
        if tz is not None:
            s, e = s.tz_localize(tz), e.tz_localize(tz)
        live = b[(b["onboarding_date"] <= e)
                 & (b["actual_exit_date"].isna() | (b["actual_exit_date"] >= s))]
        move_ins = int(((b["onboarding_date"] >= s) & (b["onboarding_date"] <= e)).sum())
        rows.append({
            "billing_period": m,
            "occupied_beds": int(live["_bed"].nunique()),
            "active_tenants_occ": int(live["full_name"].nunique()),
            "move_ins": move_ins,
            "move_outs": int(((b["actual_exit_date"] >= s)
                              & (b["actual_exit_date"] <= e)).sum()),
            "notice_count": int(((b["notice_date"] >= s)
                                 & (b["notice_date"] <= e)).sum()),
            "new_bookings": move_ins,          # no booking_date -> onboarding proxy
            "avg_monthly_rental": float(live["monthly_rental"].mean()),
        })
    df = pd.DataFrame(rows)
    df["occupancy_pct"] = (df["occupied_beds"] / config.TOTAL_BEDS * 100).round(2)
    df["vacant_beds"] = config.TOTAL_BEDS - df["occupied_beds"]
    return df


def _monthly_cash_collected(inv: pd.DataFrame, payments) -> pd.Series:
    """Cash received per calendar month from the payments ledger, keeping only
    payments that link to a real invoice via tenant_allotment_id (primary) or
    tenant_id (fallback). Returns a Period-indexed amount series."""
    if payments is None or not len(payments) or "amount_paid" not in payments.columns:
        return pd.Series(dtype=float)
    alot = set(inv["allotment_id"].dropna().astype(str)) if "allotment_id" in inv else set()
    tset = set(inv["tenant_id"].dropna().astype(str)) if "tenant_id" in inv else set()
    p = payments.copy()
    ok = pd.Series(False, index=p.index)
    if "tenant_allotment_id" in p.columns:                 # primary join key
        ok = ok | p["tenant_allotment_id"].astype(str).isin(alot)
    if "tenant_id" in p.columns:                           # fallback join key
        ok = ok | p["tenant_id"].astype(str).isin(tset)
    p = p[ok]
    pd_date = pd.to_datetime(p["payment_date"], utc=True, errors="coerce")
    period = pd_date.dt.tz_localize(None).dt.to_period("M")
    return p.assign(_p=period).groupby("_p")["amount_paid"].sum()


# --------------------------------------------------------------------------- #
# Portfolio monthly KPI table
# --------------------------------------------------------------------------- #
def property_month(inv: pd.DataFrame, elec: pd.DataFrame,
                   bookings: pd.DataFrame | None = None,
                   expenses: pd.DataFrame | None = None,
                   payments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Portfolio-level monthly KPI table (revenue, collections, occupancy, ...)."""
    rev = (inv.groupby("billing_period")
              .agg(revenue=("total_amount", "sum"),
                   rent=("rent_amount", "sum"),
                   electricity_billed=("electricity_amount", "sum"),
                   invoices=("invoice_id", "count"),
                   active_tenants=("tenant_id", "nunique"),
                   unpaid=("is_unpaid", "sum"),
                   collections=("amount_paid", "sum"),
                   outstanding_balance=("balance", "sum"),
                   average_rent=("rent_amount", "mean"))
              .reset_index())
    rev["collection_rate"] = 1 - rev["unpaid"] / rev["invoices"]
    rev["revenue_at_risk"] = (
        inv.assign(risk=inv["total_amount"] * inv["is_unpaid"])
           .groupby("billing_period")["risk"].sum().values)
    rev["arpu"] = rev["revenue"] / rev["active_tenants"]
    rev["collection_rate_amount"] = np.where(
        rev["revenue"] > 0, rev["collections"] / rev["revenue"], 0)

    e = (elec.groupby("billing_period")
             .agg(units=("units_consumed", "sum"),
                  elec_cost=("amount", "sum")).reset_index())
    out = rev.merge(e, on="billing_period", how="left")
    out["units"] = out["units"].fillna(0)
    out["elec_cost"] = out["elec_cost"].fillna(0)

    # Monthly expenses (real ledger, keyed by billing_month).
    if expenses is not None and len(expenses) and "billing_month" in expenses.columns:
        ex = expenses.copy()
        ex["billing_period"] = utils.billing_month_to_period(ex["billing_month"])
        exm = (ex.dropna(subset=["billing_period"])
                 .groupby("billing_period")["amount"].sum().rename("monthly_expenses"))
        out = out.merge(exm, on="billing_period", how="left")
    if "monthly_expenses" not in out.columns:
        out["monthly_expenses"] = 0.0
    out["monthly_expenses"] = out["monthly_expenses"].fillna(0.0)
    out["net_revenue"] = out["revenue"] - out["monthly_expenses"]

    # Cash actually received per month (payments ledger, allotment/tenant join).
    cash = _monthly_cash_collected(inv, payments)
    out["monthly_cash_collected"] = out["billing_period"].map(cash).fillna(0.0)

    # Real monthly occupancy from booking history.
    if bookings is not None:
        occ = occupancy_month(bookings, list(out["billing_period"]))
        out = out.merge(occ, on="billing_period", how="left")

    out["revenue_per_occupied_bed"] = np.where(
        out.get("occupied_beds", 0) > 0,
        out["revenue"] / out.get("occupied_beds", np.nan), 0.0)

    out["month_num"] = out["billing_period"].dt.month
    out["year"] = out["billing_period"].dt.year
    out = out.sort_values("billing_period").reset_index(drop=True)

    for lag in (1, 2, 3, 12):
        out[f"revenue_lag{lag}"] = out["revenue"].shift(lag)
    out["revenue_roll3"] = out["revenue"].rolling(3).mean()
    out["revenue_mom"] = out["revenue"].pct_change()
    out["tenants_lag1"] = out["active_tenants"].shift(1)
    for col in ("occupancy_pct", "occupied_beds", "active_tenants_occ",
                "move_ins", "move_outs", "notice_count", "new_bookings",
                "avg_monthly_rental"):
        if col in out.columns:
            out[f"{col}_lag1"] = out[col].shift(1)
    return out


# --------------------------------------------------------------------------- #
# Per-tenant billing profile
# --------------------------------------------------------------------------- #
def tenant_features(inv: pd.DataFrame) -> pd.DataFrame:
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
    out["tenure_months"] = (
        (out["last_month"].dt.year - out["first_month"].dt.year) * 12
        + (out["last_month"].dt.month - out["first_month"].dt.month) + 1)
    out["unpaid_ratio"] = out["unpaid_count"] / out["invoices"]
    out["avg_monthly_billing"] = out["total_billed"] / out["invoices"]
    paid = (inv.assign(paid_amt=inv["total_amount"] * (1 - inv["is_unpaid"]))
               .groupby("tenant_id")["paid_amt"].sum())
    out["ltv_paid"] = out["tenant_id"].map(paid)
    return out


# --------------------------------------------------------------------------- #
# Current bed view (from the derived snapshot)
# --------------------------------------------------------------------------- #
def bed_features(beds: pd.DataFrame) -> pd.DataFrame:
    df = beds.copy()
    df["is_occupied"] = (df["bed_lifecycle_status"] == "occupied").astype(int)
    df["on_notice"] = (df["bed_lifecycle_status"] == "notice").astype(int)
    df["is_vacant"] = (df["bed_lifecycle_status"] == "vacant").astype(int)
    return df


# --------------------------------------------------------------------------- #
# NEW: per-apartment features (occupancy / rent / electricity / active notices)
# --------------------------------------------------------------------------- #
def apartment_features(beds: pd.DataFrame, bookings: pd.DataFrame,
                       elec: pd.DataFrame) -> pd.DataFrame:
    b = beds.copy()
    b["occupied"] = (b["bed_lifecycle_status"] == "occupied").astype(int)
    b["vacant"] = (b["bed_lifecycle_status"] == "vacant").astype(int)
    apt = (b.groupby("apartment_code")
             .agg(total_beds=("bed_code", "count"),
                  occupied_beds=("occupied", "sum"),
                  vacant_beds=("vacant", "sum"),
                  avg_rent=("current_rate", "mean")).reset_index())
    apt["occupancy_pct"] = (apt["occupied_beds"] / apt["total_beds"] * 100).round(1)
    apt["block"] = _block(apt["apartment_code"])

    # Active notices per apartment (current, not historical): a notice is active
    # when it has not yet exited (actual_exit null) and its exit date is >= now.
    now = pd.Timestamp.now(tz="UTC")
    bk = bookings.copy()
    active = bk[(bk["notice_date"].notna())
                & (bk["actual_exit_date"].isna())
                & (bk["estimated_exit_date"] >= now)]
    an = (active.groupby("apartment_code").size().rename("active_notices"))
    apt = apt.merge(an, on="apartment_code", how="left")
    apt["active_notices"] = apt["active_notices"].fillna(0).astype(int)

    # Electricity per apartment (latest readings).
    el = (elec.groupby("apartment_code")
              .agg(elec_units=("units_consumed", "sum"),
                   elec_amount=("amount", "sum")).reset_index())
    apt = apt.merge(el, on="apartment_code", how="left")
    apt["elec_units"] = apt["elec_units"].fillna(0.0)
    apt["elec_amount"] = apt["elec_amount"].fillna(0.0)
    return apt.sort_values("occupancy_pct", ascending=False).reset_index(drop=True)


def build_all(cleaned: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    cleaned = cleaned or preprocessing.clean_all()
    return {
        "invoice_features": invoice_features(cleaned["invoices"]),
        "electricity_features": electricity_features(cleaned["electricity"]),
        "property_month": property_month(
            cleaned["invoices"], cleaned["electricity"], cleaned.get("bookings"),
            cleaned.get("expenses"), cleaned.get("payments")),
        "tenant_features": tenant_features(cleaned["invoices"]),
        "bed_features": bed_features(cleaned["beds_snapshot"]),
        "apartment_features": apartment_features(
            cleaned["beds_snapshot"], cleaned["bookings"], cleaned["electricity"]),
    }


if __name__ == "__main__":
    feats = build_all()
    for name, df in feats.items():
        out = config.OUT_DIR / f"feat_{name}.csv"
        df.to_csv(out, index=False)
        print(f"{name:22} {df.shape} -> {out.name}")
