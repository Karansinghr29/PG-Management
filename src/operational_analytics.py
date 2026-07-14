"""Operational analytics on the previously-unused real datasets.

Pure pandas data-prep for the new dashboard tabs (assets, available beds,
maintenance, apartment performance, notice & exit). Returns DataFrames only -
Plotly rendering stays in dashboard.py so styling is consistent.

Every function uses only real columns. Metrics that the data cannot support
(warranty, assets-by-apartment, apartment revenue, notice reasons, ...) are
simply not produced; see reports/dataset_audit.md for the reasons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _block(apartment_code: pd.Series) -> pd.Series:
    """Block / wing = leading letters of the apartment code (real, deterministic).
    Floor is NOT reliably derivable, so we expose block only."""
    return apartment_code.astype("string").str.extract(r"^([A-Za-z]+)")[0]


# --------------------------------------------------------------------------- #
# Module 1 - Assets
# --------------------------------------------------------------------------- #
def assets_summary(assets: pd.DataFrame) -> dict:
    a = assets.copy()
    out = {
        "total": int(len(a)),
        "by_category": a["category"].value_counts().reset_index(),
        "by_type": a["asset_type"].value_counts().head(15).reset_index(),
        "by_status": a["status"].value_counts().reset_index(),
        "by_condition": a["condition"].value_counts().reset_index(),
    }
    for k in ("by_category", "by_type", "by_status", "by_condition"):
        out[k].columns = [k.replace("by_", ""), "count"]
    # Purchase timeline only for rows that actually have a purchase_date (~18%).
    pd_col = pd.to_datetime(a.get("purchase_date"), errors="coerce")
    have = pd_col.notna()
    out["purchase_coverage"] = int(have.sum())
    if have.any():
        tl = (a.loc[have].assign(month=pd_col[have].dt.to_period("M").astype(str))
                .groupby("month")
                .agg(assets=("asset_code", "count"),
                     spend=("purchase_price", "sum")).reset_index())
        out["purchase_timeline"] = tl
    else:
        out["purchase_timeline"] = None
    out["table"] = a[["asset_code", "asset_type", "category", "condition",
                      "status"]]
    return out


# --------------------------------------------------------------------------- #
# Module 2 - Available beds
# --------------------------------------------------------------------------- #
def bed_availability(beds: pd.DataFrame) -> dict:
    b = beds.copy()
    b["is_vacant"] = (b["bed_lifecycle_status"] == "vacant").astype(int)
    b["block"] = _block(b["apartment_code"])
    total = len(b)
    vacant = b[b["is_vacant"] == 1]
    block = (b.groupby("block")
               .agg(total=("bed_code", "count"),
                    vacant=("is_vacant", "sum")).reset_index())
    block["vacancy_pct"] = (block["vacant"] / block["total"] * 100).round(1)
    apt = (b.groupby("apartment_code")
             .agg(total=("bed_code", "count"),
                  vacant=("is_vacant", "sum")).reset_index())
    apt["vacancy_pct"] = (apt["vacant"] / apt["total"] * 100).round(1)
    return {
        "total_beds": int(total),
        "vacant_beds": int(vacant.shape[0]),
        "occupancy_pct": round((1 - vacant.shape[0] / total) * 100, 1),
        "vacant_revenue_opportunity": float(vacant["current_rate"].sum()),
        "by_block": block.sort_values("vacancy_pct", ascending=False),
        "by_apartment": apt.sort_values("vacancy_pct", ascending=False),
        "vacant_table": vacant[["apartment_code", "bed_code", "bed_type",
                                "toilet_type", "gender_allowed", "current_rate"]],
        "lifecycle": b["bed_lifecycle_status"].value_counts().reset_index()
                      .set_axis(["status", "count"], axis=1),
    }


# --------------------------------------------------------------------------- #
# Module 3 - Maintenance performance
# --------------------------------------------------------------------------- #
def maintenance_summary(tickets: pd.DataFrame) -> dict:
    t = tickets.copy()
    created = pd.to_datetime(t["created_at"], errors="coerce", utc=True)
    t["month"] = created.dt.to_period("M").astype(str)
    closed_mask = t["status"].eq("closed")
    monthly = (t.groupby("month")
                 .agg(tickets=("ticket_number", "count")).reset_index())
    apt = (t.groupby("apartment_code")
             .agg(complaints=("ticket_number", "count")).reset_index()
             .sort_values("complaints", ascending=False))
    res = t["resolution_hours"].dropna()
    res = res[(res >= 0)]
    return {
        "total": int(len(t)),
        "open": int((~closed_mask).sum()),
        "closed": int(closed_mask.sum()),
        "avg_resolution_hours": float(res.mean()) if len(res) else float("nan"),
        "sla_breached": int(t["sla_breached"].sum()) if "sla_breached" in t
        else None,
        "sla_breach_pct": round(t["sla_breached"].mean() * 100, 1)
        if "sla_breached" in t else None,
        "by_status": t["status"].value_counts().reset_index()
                      .set_axis(["status", "count"], axis=1),
        "by_priority": t["priority"].value_counts().reset_index()
                        .set_axis(["priority", "count"], axis=1),
        "by_issue": t["issue_type"].value_counts().reset_index()
                     .set_axis(["issue_type", "count"], axis=1),
        "by_apartment": apt,
        "monthly": monthly,
        "resolution_series": res,
    }


# --------------------------------------------------------------------------- #
# Module 4 - Apartment performance (only real, apartment-keyed metrics)
# --------------------------------------------------------------------------- #
def apartment_performance(electricity, tickets, beds) -> pd.DataFrame:
    """Per-apartment metrics from datasets that genuinely carry apartment_code.
    Revenue / collection are intentionally absent: invoices have no apartment
    code (see audit)."""
    elec = (electricity.groupby("apartment_code")
                       .agg(elec_cost=("amount", "sum"),
                            avg_units=("units_consumed", "mean")).reset_index())
    comp = (tickets.groupby("apartment_code")
                   .agg(complaints=("ticket_number", "count")).reset_index())
    b = beds.copy()
    b["is_vacant"] = (b["bed_lifecycle_status"] == "vacant").astype(int)
    vac = (b.groupby("apartment_code")
             .agg(beds=("bed_code", "count"),
                  vacant=("is_vacant", "sum")).reset_index())

    df = (elec.merge(comp, on="apartment_code", how="outer")
              .merge(vac, on="apartment_code", how="outer"))
    for c in ["elec_cost", "avg_units", "complaints", "beds", "vacant"]:
        df[c] = df[c].fillna(0)

    # Health score (0-100, higher = healthier). Transparent percentile blend:
    # fewer complaints, lower vacancy, lower electricity deviation are better.
    def inv_rank(s):                       # lower value -> higher score
        return (1 - s.rank(pct=True)) * 100
    df["health_score"] = (
        0.45 * inv_rank(df["complaints"])
        + 0.35 * inv_rank(df["vacant"])
        + 0.20 * inv_rank(df["elec_cost"])).round(1)
    df["complaint_rank"] = df["complaints"].rank(ascending=False).astype(int)
    df["elec_rank"] = df["elec_cost"].rank(ascending=False).astype(int)
    return df.sort_values("health_score", ascending=False)


# --------------------------------------------------------------------------- #
# Module 5 - Notice & exit
# --------------------------------------------------------------------------- #
def notice_analytics(notices: pd.DataFrame) -> dict:
    n = notices.copy()
    n["notice_date"] = pd.to_datetime(n["notice_date"], errors="coerce", utc=True)
    n["estimated_exit_date"] = pd.to_datetime(n["estimated_exit_date"],
                                              errors="coerce", utc=True)
    n["notice_month"] = n["notice_date"].dt.to_period("M").astype(str)
    n["exit_month"] = n["estimated_exit_date"].dt.to_period("M").astype(str)
    now = pd.Timestamp.now(tz="UTC")
    upcoming = n[n["estimated_exit_date"] >= now]
    monthly = (n.groupby("notice_month")
                 .agg(notices=("full_name", "count"),
                      revenue_impact=("monthly_rental", "sum")).reset_index())
    apt = (n.groupby("apartment_code")
             .agg(notices=("full_name", "count"),
                  revenue_impact=("monthly_rental", "sum")).reset_index()
             .sort_values("notices", ascending=False))
    exit_month = (n.groupby("exit_month")
                    .agg(exits=("full_name", "count"),
                         revenue_impact=("monthly_rental", "sum")).reset_index())
    return {
        "total_notices": int(len(n)),
        "upcoming_exits": int(len(upcoming)),
        "monthly_revenue_impact": float(n["monthly_rental"].sum()),
        "avg_notice_days": float(n.get("notice_period_days",
                                       pd.Series(dtype=float)).mean()),
        "monthly": monthly,
        "by_apartment": apt,
        "exit_month": exit_month.sort_values("exit_month"),
        "upcoming_table": upcoming[["full_name", "apartment_code", "bed_code",
                                    "estimated_exit_date", "monthly_rental"]]
                          .sort_values("estimated_exit_date"),
    }


# --------------------------------------------------------------------------- #
# Module 6 - Financial-year REVENUE analytics (real billed revenue only, no ML,
#            no collection/payment inference — there is no real payments table)
# --------------------------------------------------------------------------- #
def _financial_year_bounds(latest: pd.Period):
    """April->March financial year CONTAINING `latest`. Auto, never hardcoded."""
    start_year = latest.year if latest.month >= 4 else latest.year - 1
    fy_start = pd.Period(f"{start_year}-04", freq="M")
    fy_end = pd.Period(f"{start_year + 1}-03", freq="M")
    return fy_start, fy_end, f"{start_year}-{str(start_year + 1)[-2:]}"


def _fy_label_of(p: pd.Period) -> str:
    sy = p.year if p.month >= 4 else p.year - 1
    return f"{sy}-{str(sy + 1)[-2:]}"


def financial_year_revenue(invoices: pd.DataFrame) -> dict:
    """Current financial year BILLED revenue per month. Real invoice revenue only
    (sum of total_amount) — no collection, no payment inference. FY auto-detected
    from the latest billing_period; previous years excluded."""
    inv = invoices.copy()
    latest = inv["billing_period"].max()
    fy_start, fy_end, label = _financial_year_bounds(latest)
    fy = inv[(inv["billing_period"] >= fy_start) & (inv["billing_period"] <= fy_end)]
    monthly = (fy.groupby("billing_period")["total_amount"].sum()
                 .reset_index().sort_values("billing_period"))
    monthly["month"] = monthly["billing_period"].astype(str)
    monthly = monthly.rename(columns={"total_amount": "revenue"})[["month", "revenue"]]
    return {
        "fy_label": label, "fy_start": str(fy_start), "fy_end": str(fy_end),
        "revenue": float(fy["total_amount"].sum()),
        "n_invoices": int(len(fy)),
        "monthly": monthly,
    }


def monthly_revenue_trend(invoices: pd.DataFrame) -> pd.DataFrame:
    """All-time monthly BILLED revenue (real invoice totals only)."""
    inv = invoices.copy()
    m = (inv.groupby("billing_period")["total_amount"].sum()
            .reset_index().sort_values("billing_period"))
    m["month"] = m["billing_period"].astype(str)
    return m.rename(columns={"total_amount": "revenue"})[["month", "revenue"]]


def revenue_by_financial_year(invoices: pd.DataFrame) -> pd.DataFrame:
    """Total BILLED revenue per financial year (Apr->Mar), all years present in
    the data. Marks the latest FY as in-progress when it has < 12 months. Real
    invoice revenue only."""
    inv = invoices.copy()
    inv["fy"] = inv["billing_period"].map(_fy_label_of)
    g = (inv.groupby("fy")
            .agg(revenue=("total_amount", "sum"),
                 months=("billing_period", "nunique"),
                 invoices=("invoice_id", "count")).reset_index()
            .sort_values("fy"))
    g["in_progress"] = g["months"] < 12
    return g


# --------------------------------------------------------------------------- #
# Module 7 - The single kept operational alert: vacant apartment using power
# --------------------------------------------------------------------------- #
def vacant_apartment_power_alerts(cleaned: dict) -> pd.DataFrame:
    """The ONLY operational alert kept in the app.

    Rule:  apartment status is Vacant  AND  units_consumed > 0  ->  one alert.
    "Vacant" = every bed in the apartment is vacant in the current snapshot.
    Uses the latest electricity billing month. No other anomaly detection —
    no billing, invoice, exit, duplicate or statistical-outlier checks.

    Returns rows: apartment_code, status, units_consumed, expected_units,
    billing_month.
    """
    cols = ["apartment_code", "status", "units_consumed", "expected_units",
            "billing_month"]
    el = cleaned.get("electricity")
    bs = cleaned.get("beds_snapshot")
    if el is None or bs is None or not len(el) or not len(bs):
        return pd.DataFrame(columns=cols)

    occ = (bs.assign(vac=(bs["bed_lifecycle_status"] == "vacant").astype(int))
             .groupby("apartment_code")
             .agg(beds=("bed_code", "count"), vac=("vac", "sum")).reset_index())
    vacant = set(occ.loc[occ["beds"] == occ["vac"], "apartment_code"])
    if not vacant:
        return pd.DataFrame(columns=cols)

    last = el["billing_period"].max()
    hit = el[(el["billing_period"] == last)
             & (el["apartment_code"].isin(vacant))
             & (el["units_consumed"] > 0)].copy()
    if not len(hit):
        return pd.DataFrame(columns=cols)
    hit["status"] = "Vacant Apartment Using Power"
    hit["expected_units"] = 0
    return hit[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Module 8 - Business insights (real data only; honest about what's available)
# --------------------------------------------------------------------------- #
def business_insights(cleaned: dict, feats: dict) -> dict:
    """Actionable insights for the PG owner from REAL data only.

    Per-apartment occupancy uses the CURRENT bed snapshot — the dataset has no
    per-apartment occupancy history (bookings carry UUID apartment ids with no
    link to apartment_code), so anything apartment-level is a current snapshot.
    Portfolio peak-month occupancy uses the real booking-based monthly series.
    Nothing is invented: metrics that cannot be computed honestly are returned
    as None / "Not Available".
    """
    NA = "Not Available"
    bs = cleaned.get("beds_snapshot")
    pm = feats.get("property_month")
    out = {
        "most_booked": None, "peak_month": None,
        "vacant_beds": pd.DataFrame(), "rent_opportunity": pd.DataFrame(),
        "low_demand": pd.DataFrame(),
        "apartment_history_available": False,  # per-apartment history not in data
    }
    if bs is None or not len(bs):
        return out

    b = bs.copy()
    b["occupied"] = (b["bed_lifecycle_status"] == "occupied").astype(int)
    b["is_vacant"] = (b["bed_lifecycle_status"] == "vacant").astype(int)
    apt = (b.groupby("apartment_code")
             .agg(total_beds=("bed_code", "count"),
                  occupied=("occupied", "sum"),
                  vacant=("is_vacant", "sum")).reset_index())
    apt["occupancy_pct"] = (apt["occupied"] / apt["total_beds"] * 100).round(1)
    apt["block"] = _block(apt["apartment_code"])

    # 1) Most booked apartment (highest current occupancy; tie -> most beds).
    top = apt.sort_values(["occupancy_pct", "occupied"], ascending=False).iloc[0]
    out["most_booked"] = {
        "apartment": str(top["apartment_code"]), "block": str(top["block"]),
        "occupancy_pct": float(top["occupancy_pct"]),
        "active_beds": int(top["occupied"])}

    # 2) Peak occupancy month (real booking-based monthly history).
    if pm is not None and "occupancy_pct" in pm.columns and \
            pm["occupancy_pct"].notna().any():
        o = pm.dropna(subset=["occupancy_pct"])
        r = o.loc[o["occupancy_pct"].idxmax()]
        out["peak_month"] = {
            "month": str(r["billing_period"]),
            "occupancy_pct": float(r["occupancy_pct"]),
            "occupied_beds": int(r["occupied_beds"]),
            "total_beds": int(config.TOTAL_BEDS)}

    # 3) Currently vacant beds. No "vacant-since" date exists -> duration = NA.
    vac = (b[b["is_vacant"] == 1][["apartment_code", "bed_code",
                                   "bed_lifecycle_status"]].copy())
    vac["duration"] = NA
    out["vacant_beds"] = (vac.rename(columns={
        "apartment_code": "Apartment", "bed_code": "Bed Code",
        "bed_lifecycle_status": "Current Status",
        "duration": "Estimated Vacant Duration"}).reset_index(drop=True))

    # 4) Rent increase opportunity — current occupancy > 95% (recommendation only).
    high = (apt[apt["occupancy_pct"] > 95]
            .sort_values("occupancy_pct", ascending=False))
    ro = high[["apartment_code", "block", "occupancy_pct", "occupied"]].copy()
    ro["Recommendation"] = ("Apartment is currently fully occupied. If this "
                            "occupancy level continues over future months, consider "
                            "reviewing rent pricing.")
    out["rent_opportunity"] = ro.rename(columns={
        "apartment_code": "Apartment", "block": "Block",
        "occupancy_pct": "Occupancy %", "occupied": "Active Beds"}) \
        .reset_index(drop=True)

    # 5) Low demand apartments — most vacant beds.
    low = (apt[apt["vacant"] > 0].sort_values("vacant", ascending=False)
           [["apartment_code", "vacant", "occupancy_pct"]].copy())
    out["low_demand"] = low.rename(columns={
        "apartment_code": "Apartment", "vacant": "Vacant Beds",
        "occupancy_pct": "Occupancy %"}).reset_index(drop=True)
    return out


if __name__ == "__main__":
    from src import preprocessing
    c = preprocessing.clean_all()
    print("assets:", assets_summary(c["assets"])["total"],
          "purchase-cov:", assets_summary(c["assets"])["purchase_coverage"])
    ba = bed_availability(c["beds_snapshot"])
    print("beds vacant:", ba["vacant_beds"], "opp Rs:", ba["vacant_revenue_opportunity"])
    ms = maintenance_summary(c["tickets"])
    print("tickets open/closed:", ms["open"], ms["closed"],
          "sla%:", ms["sla_breach_pct"])
    ap = apartment_performance(c["electricity"], c["tickets"], c["beds_snapshot"])
    print("apartments scored:", len(ap), "top:", ap.iloc[0]["apartment_code"])
    na = notice_analytics(c["notices"])
    print("notices:", na["total_notices"], "upcoming:", na["upcoming_exits"],
          "rev impact:", na["monthly_revenue_impact"])
