"""Step 2 - cleaning. Turns raw uploads into tidy, typed, de-duplicated tables.

Every fix here is driven by a concrete issue found during data discovery
(see reports/report.md, "Data Quality"). Run standalone to materialise the
cleaned CSVs under outputs/, or import `clean_all()` from other modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402


def clean_invoices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["billing_period"] = utils.billing_month_to_period(df["billing_month"])
    # Negative electricity is impossible -> clip to 0.
    df["electricity_amount"] = df["electricity_amount"].clip(lower=0)
    # Recompute total to fix the 90 rows where components don't sum.
    df["total_amount"] = (df["rent_amount"] + df["electricity_amount"]
                          + df["other_charges"])
    df["is_unpaid"] = df["is_unpaid"].astype(int)
    return df


def clean_electricity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["billing_period"] = utils.billing_month_to_period(df["billing_month"])
    # One negative reading is a data error -> clip.
    df["units_consumed"] = df["units_consumed"].clip(lower=0)
    df["amount"] = df["amount"].clip(lower=0)
    return df


def clean_beds_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.normalise_case(df.copy(), config.CASE_NORMALISE["beds_snapshot"])
    return df


def clean_beds_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.normalise_case(df.copy(), config.CASE_NORMALISE["beds_catalog"])
    df = df.drop_duplicates()               # removes the 32 exact dupes
    return df


def clean_notices(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.parse_dates(df.copy(), config.DATE_COLS["notices"])
    df["notice_period_days"] = (
        (df["estimated_exit_date"] - df["notice_date"]).dt.days
    )
    return df


def clean_assets(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.drop_dead_cols(df.copy(), config.DEAD_COLS["assets"])
    df = utils.parse_dates(df, ["purchase_date"])
    return df


def clean_meters(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.drop_dead_cols(df.copy(), config.DEAD_COLS["meters"])
    df = df.drop_duplicates()               # removes the 1 dupe
    return df


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.parse_dates(df.copy(), config.DATE_COLS["tickets"])
    df = utils.normalise_case(df, config.CASE_NORMALISE["tickets"])  # High->high
    df = utils.drop_dead_cols(df, config.DEAD_COLS["tickets"])
    df["resolution_hours"] = (
        (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600
    )
    df["sla_breached"] = (
        (df["resolved_at"] > df["sla_deadline"]).fillna(False).astype(int)
    )
    return df


def clean_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Stay/booking history — parse the date fields used for occupancy."""
    df = utils.drop_dead_cols(df.copy(), config.DEAD_COLS["bookings"])
    df = utils.parse_dates(df, config.DATE_COLS["bookings"])
    df["staying_status"] = df["staying_status"].astype("string").str.strip()
    return df


CLEANERS = {
    "invoices": clean_invoices,
    "bookings": clean_bookings,
    "electricity": clean_electricity,
    "beds_snapshot": clean_beds_snapshot,
    "beds_catalog": clean_beds_catalog,
    "notices": clean_notices,
    "assets": clean_assets,
    "meters": clean_meters,
    "tickets": clean_tickets,
}


def clean_all(raw: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    raw = raw or utils.load_all_raw()
    return {name: CLEANERS[name](raw[name]) for name in CLEANERS}


if __name__ == "__main__":
    cleaned = clean_all()
    for name, df in cleaned.items():
        out = config.OUT_DIR / f"clean_{name}.csv"
        df.to_csv(out, index=False)
        print(f"{name:14} -> {out.name:26} {df.shape}")
