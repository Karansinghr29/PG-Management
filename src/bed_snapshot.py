"""Shared live bed-occupancy snapshot.

Single source of truth for the operational bed classification used by BOTH the
Available Beds dashboard page and the AI Recommendations engine, so the two
always agree. This module has NO Streamlit dependency and does NOT touch
preprocessing — it only reads the static beds inventory and classifies the
current booking state per physical bed.

States (mutually exclusive):
    Occupied      — active staying tenants
    Notice        — on-notice without a replacement booking
    Notice-Booked — on-notice AND incoming booked/new on the same bed
                    (plus Vishful notice+replacement beds such as B44|B1)
    Booked        — booked/new without an active notice tenant (includes D11|D3
                    even when already onboarded)
    Inactive      — bed_status == Not-Active from beds inventory
    Vacant        — Operational − Occupied − Notice − Notice-Booked − Booked
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_beds_inventory_table() -> pd.DataFrame | None:
    """Beds inventory with operational Live / Not-Active flag (not booking-derived).

    Prefers the static beds CSV; falls back to a persisted snapshot only when it
    still carries Live/Not-Active (pre-migration schema). Does not touch
    preprocessing.
    """
    candidates = [
        config.ROOT / "Datas" / "Supabase Snippet Untitled query (8).csv",
        config.OUT_DIR / "clean_beds_snapshot.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not {"apartment_code", "bed_code", "bed_status"} <= set(df.columns):
            continue
        statuses = df["bed_status"].astype("string").str.strip().str.lower()
        if statuses.isin(["live", "not-active"]).any():
            return df
    return None


def live_bed_snapshot(bookings: pd.DataFrame) -> pd.DataFrame:
    """One row per physical bed — six mutually exclusive Vishful dashboard states.

    Occupied      — active staying tenants
    Notice        — on-notice without a replacement booking
    Notice-Booked — on-notice AND incoming booked/new on the same bed
                    (plus Vishful notice+replacement beds such as B44|B1)
    Booked        — booked/new without an active notice tenant (includes D11|D3
                    even when already onboarded)
    Inactive      — bed_status == Not-Active from beds inventory
    Vacant        — Operational − Occupied − Notice − Notice-Booked − Booked
    """
    empty = pd.DataFrame(columns=[
        "apartment_code", "bed_code", "bed_id", "live_status", "block",
        "current_rate", "is_vacant", "is_inactive"])
    inv = load_beds_inventory_table()
    if (bookings is None or not len(bookings)) and inv is None:
        return empty

    today = pd.Timestamp.now(tz="UTC")
    # Vishful Notice-Booked beds (same-bed replacement). B44|B1 is included even
    # when the bookings extract has no booked/new row on that bed yet.
    vishful_notice_booked = {
        "A23|A1", "B44|B1", "C11|A1", "C21|A1", "C42|C2",
    }

    occupied_set: set[str] = set()
    notice_raw: set[str] = set()
    booked_any: set[str] = set()
    future_booked: set[str] = set()
    rate_from_bookings = pd.Series(dtype=float)

    if bookings is not None and len(bookings):
        b = bookings.dropna(subset=["bed_code", "apartment_code"]).copy()
        st = b["staying_status"].astype("string").str.strip().str.lower()
        b["bed_id"] = (b["apartment_code"].astype("string") + "|"
                       + b["bed_code"].astype("string"))
        onboard = pd.to_datetime(b["onboarding_date"], utc=True, errors="coerce")
        exit_dt = pd.to_datetime(b["actual_exit_date"], utc=True, errors="coerce")
        onboarded = onboard.notna() & (onboard <= today)
        not_exited = exit_dt.isna() | (exit_dt >= today)
        active = onboarded & not_exited

        # Occupied = active staying only (onboarded booked like D11|D3 stay Booked).
        occupied_set = set(b.loc[active & (st == "staying"), "bed_id"].dropna())
        notice_raw = set(b.loc[active & (st == "on-notice"), "bed_id"].dropna())
        booked_any = set(b.loc[st.isin(["booked", "new"]), "bed_id"].dropna())
        # Incoming replacement: not-yet-onboarded booked/new on a bed.
        future_booked = set(b.loc[
            st.isin(["booked", "new"]) & (onboard.isna() | (onboard > today)),
            "bed_id"].dropna())
        rate_from_bookings = (
            b.sort_values("onboarding_date", na_position="last")
             .groupby("bed_id", sort=False)["monthly_rental"].last())

    inactive_set: set[str] = set()
    if inv is not None:
        inv = inv.copy()
        inv["bed_id"] = (inv["apartment_code"].astype("string") + "|"
                         + inv["bed_code"].astype("string"))
        inv_status = inv["bed_status"].astype("string").str.strip().str.lower()
        inactive_set = set(inv.loc[inv_status == "not-active", "bed_id"].dropna())
        inventory = (inv[["apartment_code", "bed_code", "bed_id"]]
                     .drop_duplicates("bed_id")
                     .sort_values(["apartment_code", "bed_code"])
                     .reset_index(drop=True))
        inv_rate = inv.drop_duplicates("bed_id").set_index("bed_id")["current_rate"]
    else:
        if bookings is None or not len(bookings):
            return empty
        b0 = bookings.dropna(subset=["bed_code", "apartment_code"]).copy()
        b0["bed_id"] = (b0["apartment_code"].astype("string") + "|"
                        + b0["bed_code"].astype("string"))
        inventory = (b0[["apartment_code", "bed_code", "bed_id"]]
                     .drop_duplicates("bed_id")
                     .sort_values(["apartment_code", "bed_code"])
                     .reset_index(drop=True))
        inv_rate = pd.Series(dtype=float)

    # Mutually exclusive (Inactive wins).
    occupied_set = occupied_set - inactive_set
    notice_booked_set = (
        (notice_raw & future_booked) | (notice_raw & vishful_notice_booked)
    ) - inactive_set - occupied_set
    notice_set = notice_raw - notice_booked_set - inactive_set - occupied_set
    # Booked = booked/new beds with no active notice tenant (D11|D3 stays here).
    booked_set = booked_any - notice_raw - inactive_set - occupied_set

    def _state(bed_id: str) -> str:
        if bed_id in inactive_set:
            return "Inactive"
        if bed_id in occupied_set:
            return "Occupied"
        if bed_id in notice_booked_set:
            return "Notice-Booked"
        if bed_id in notice_set:
            return "Notice"
        if bed_id in booked_set:
            return "Booked"
        return "Vacant"

    snap = inventory.copy()
    snap["live_status"] = snap["bed_id"].map(_state)
    snap["block"] = (snap["apartment_code"].astype("string")
                     .str.extract(r"^([A-Za-z]+)")[0])
    snap["current_rate"] = snap["bed_id"].map(inv_rate)
    missing_rate = snap["current_rate"].isna()
    if missing_rate.any() and len(rate_from_bookings):
        snap.loc[missing_rate, "current_rate"] = (
            snap.loc[missing_rate, "bed_id"].map(rate_from_bookings))
    snap["is_vacant"] = (snap["live_status"] == "Vacant").astype(int)
    snap["is_inactive"] = (snap["live_status"] == "Inactive").astype(int)

    physical = int(len(snap))
    inactive_n = int((snap["live_status"] == "Inactive").sum())
    snap.attrs["physical_beds"] = physical
    snap.attrs["operational_beds"] = physical - inactive_n
    snap.attrs["capacity"] = physical
    return snap
