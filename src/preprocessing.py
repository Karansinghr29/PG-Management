"""Step 2 - cleaning. Turns the raw production uploads into tidy, typed tables.

Phase-1 migration: reads the NEW Data/ datasets. Renamed columns are bridged to
the old names the rest of the pipeline expects (e.g. invoice `id` -> `invoice_id`,
electricity `eb_amount` -> `amount`), `is_unpaid` is derived from the new invoice
status/balance, `staying_status` is normalised, and the tables that no longer
exist as files (notices, beds_snapshot, beds_catalog) are DERIVED from bookings.

Business logic downstream is intentionally NOT changed in this phase — cleaners
just reproduce the old business keys/columns from the new sources so the app can
load. Run standalone to materialise clean_*.csv under outputs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _strip_tz(s: pd.Series) -> pd.Series:
    return s.dt.tz_localize(None) if getattr(s.dtype, "tz", None) is not None else s


# --------------------------------------------------------------------------- #
# File-backed cleaners
# --------------------------------------------------------------------------- #
def clean_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """New invoices (28). Bridge id->invoice_id; derive is_unpaid from
    status/balance (there is no is_unpaid column in the new data)."""
    df = utils.parse_dates(df.copy(), config.DATE_COLS["invoices"])
    df["billing_period"] = utils.billing_month_to_period(df["billing_month"])
    if "invoice_id" not in df.columns and "id" in df.columns:
        df["invoice_id"] = df["id"]                       # renamed column bridge
    df = _numeric(df, ["rent_amount", "electricity_amount", "other_charges",
                       "total_amount", "amount_paid", "balance", "late_fee"])
    df["electricity_amount"] = df["electricity_amount"].clip(lower=0)
    # Derive is_unpaid from the real payment status / outstanding balance.
    status = (df["status"].astype("string").str.strip().str.lower()
              if "status" in df.columns else pd.Series("", index=df.index))
    balance = df["balance"].fillna(0) if "balance" in df.columns else \
        pd.Series(0, index=df.index)
    df["is_unpaid"] = ((status != "paid") | (balance > 0)).astype(int)
    return df


def clean_electricity(df: pd.DataFrame) -> pd.DataFrame:
    """New EB meter readings (26). Bridge eb_amount->amount; period from
    reading_date. (Point-in-time readings, not a monthly series — see report.)"""
    df = utils.parse_dates(df.copy(), config.DATE_COLS["electricity"])
    if "amount" not in df.columns and "eb_amount" in df.columns:
        df = df.rename(columns={"eb_amount": "amount"})
    if "reading_date" in df.columns:
        df["billing_period"] = _strip_tz(df["reading_date"]).dt.to_period("M")
    df = _numeric(df, ["units_consumed", "amount"])
    for c in ("units_consumed", "amount"):
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
    return df


def clean_meters(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.drop_dead_cols(df.copy(), config.DEAD_COLS["meters"])
    df = utils.parse_dates(df, config.DATE_COLS["electricity"])
    return df.drop_duplicates()


def clean_eb_bills(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.parse_dates(df.copy(), config.DATE_COLS["eb_bills"])
    return _numeric(df, ["bill_amount"])


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.parse_dates(df.copy(), config.DATE_COLS["tickets"])
    df = utils.normalise_case(df, config.CASE_NORMALISE["tickets"])   # High->high
    df = utils.drop_dead_cols(df, config.DEAD_COLS["tickets"])
    df["resolution_hours"] = (
        (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600)
    df["sla_breached"] = (
        (df["resolved_at"] > df["sla_deadline"]).fillna(False).astype(int))
    return df


def clean_assets(df: pd.DataFrame) -> pd.DataFrame:
    """New assets (27) — apartment_code is now populated (kept, not dropped)."""
    df = utils.parse_dates(df.copy(), config.DATE_COLS["assets"])
    return _numeric(df, ["purchase_price", "warranty_months"])


def clean_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """New bookings/stay history (22), keyed by full_name/apartment_code/bed_code
    (no UUIDs). Normalise staying_status so 'Exited'/'exited' collapse."""
    df = utils.parse_dates(df.copy(), config.DATE_COLS["bookings"])
    df = utils.normalise_case(df, config.CASE_NORMALISE["bookings"])
    return _numeric(df, ["monthly_rental", "total_due", "paid_amount",
                         "balance_due", "deposit_paid"])


def clean_tenants(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.parse_dates(df.copy(), config.DATE_COLS["tenants"])
    df = utils.normalise_case(df, config.CASE_NORMALISE["tenants"])
    return _numeric(df, ["tenant_rating", "age"])


def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    """New payments / receipts ledger (29)."""
    df = utils.parse_dates(df.copy(), config.DATE_COLS["payments"])
    return _numeric(df, ["amount_paid", "base_amount", "processing_fee"])


def clean_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """New expenses ledger (30)."""
    df = utils.parse_dates(df.copy(), config.DATE_COLS["expenses"])
    return _numeric(df, ["amount"])


# --------------------------------------------------------------------------- #
# Derived tables (no source file in the new data)
# --------------------------------------------------------------------------- #
def derive_notices(bookings: pd.DataFrame) -> pd.DataFrame:
    """Notices = bookings rows that carry a notice_date. Reproduces the old
    notices schema (full_name, apartment_code, bed_code, dates, monthly_rental)."""
    n = bookings[bookings["notice_date"].notna()].copy()
    keep = ["full_name", "phone", "notice_date", "estimated_exit_date",
            "property_name", "apartment_code", "bed_code", "monthly_rental"]
    n = n[[c for c in keep if c in n.columns]].copy()
    if {"estimated_exit_date", "notice_date"} <= set(n.columns):
        n["notice_period_days"] = (
            (n["estimated_exit_date"] - n["notice_date"]).dt.days)
    return n.reset_index(drop=True)


def _current_bed_rows(bookings: pd.DataFrame) -> pd.DataFrame:
    """Latest stay per (apartment_code, bed_code) — the current occupant/state.
    Approximated from booking history (the new data has no bed snapshot file)."""
    b = bookings.dropna(subset=["bed_code"]).copy()
    b = b.sort_values("onboarding_date")
    return b.drop_duplicates(["apartment_code", "bed_code"], keep="last")


def derive_beds_snapshot(bookings: pd.DataFrame) -> pd.DataFrame:
    """Current bed snapshot derived from bookings. Maps normalised staying_status
    to the old bed_lifecycle_status vocabulary (occupied/notice/booked/vacant).
    NOTE: only beds present in booking history appear (no separate bed inventory
    exists in the new data) — full reconstruction is a later phase."""
    cur = _current_bed_rows(bookings)
    st = cur["staying_status"].astype("string").str.strip().str.lower()
    lifecycle = st.map(config.STAYING_STATUS_MAP).fillna("vacant")
    return pd.DataFrame({
        "property": cur.get("property_name"),
        "apartment_code": cur["apartment_code"],
        "gender_allowed": pd.NA,
        "bed_code": cur["bed_code"],
        "bed_type": cur.get("bed_type"),
        "toilet_type": pd.NA,
        "bed_status": cur["staying_status"],
        "bed_lifecycle_status": lifecycle.to_numpy(),
        "current_rate": cur.get("monthly_rental"),
    }).reset_index(drop=True)


def derive_beds_catalog(bookings: pd.DataFrame) -> pd.DataFrame:
    """Bed inventory + pricing derived from bookings (latest rate per bed)."""
    cur = _current_bed_rows(bookings)
    return pd.DataFrame({
        "property_name": cur.get("property_name"),
        "apartment_code": cur["apartment_code"],
        "bed_code": cur["bed_code"],
        "bed_type": cur.get("bed_type"),
        "toilet_type": pd.NA,
        "monthly_rate": cur.get("monthly_rental"),
        "gender_allowed": pd.NA,
        "status": cur["staying_status"],
    }).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def clean_all(raw: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Clean every source into the business-named tables the app expects.
    Keeps all old keys (invoices, electricity, beds_snapshot, beds_catalog,
    notices, assets, meters, tickets, bookings) and adds the new sources
    (tenants, payments, expenses, eb_bills)."""
    raw = raw or utils.load_all_raw()
    bookings = clean_bookings(raw["bookings"])
    return {
        "invoices": clean_invoices(raw["invoices"]),
        "bookings": bookings,
        "electricity": clean_electricity(raw["electricity"]),
        "meters": clean_meters(raw["meters"]),
        "eb_bills": clean_eb_bills(raw["eb_bills"]),
        "tickets": clean_tickets(raw["tickets"]),
        "assets": clean_assets(raw["assets"]),
        "tenants": clean_tenants(raw["tenants"]),
        "payments": clean_payments(raw["payments"]),
        "expenses": clean_expenses(raw["expenses"]),
        # derived from bookings (no source file in the new data):
        "notices": derive_notices(bookings),
        "beds_snapshot": derive_beds_snapshot(bookings),
        "beds_catalog": derive_beds_catalog(bookings),
    }


if __name__ == "__main__":
    cleaned = clean_all()
    for name, df in cleaned.items():
        out = config.OUT_DIR / f"clean_{name}.csv"
        df.to_csv(out, index=False)
        print(f"{name:14} -> {out.name:28} {df.shape}")
