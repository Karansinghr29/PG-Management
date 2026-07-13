"""AI Recommendation Engine — the final business-assistant layer.

Architecture:

    Raw data
        |
    Business analytics  (src/operational_analytics.py + persisted forecast outputs)
        |            each dashboard page's single source of truth
        v
    collect_business_outputs()   -> gathers every page's STRUCTURED output
        |
    generate_recommendations()   -> summarises those outputs into actions
        |
    AI Recommendations page

The engine NEVER re-derives a page's metric. It calls the same analytics
functions the pages call (once) and reads the same persisted forecast outputs
the forecast pages display, then turns the collected numbers into prioritised
recommendations. So when any page's number changes (new data), the matching
recommendation changes automatically — no duplicated business logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import operational_analytics as ops  # noqa: E402

_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _load_csv(name: str):
    p = config.OUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _load_json(name: str):
    p = config.OUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
# Collect — one structured output per dashboard page (single source of truth)
# --------------------------------------------------------------------------- #
def _safe(fn, default=None):
    """Run one page's collector in isolation. If that page (or its data/output)
    is missing or errors, return `default` so the rest still work — this is what
    lets the engine ignore a removed page instead of crashing."""
    try:
        return fn()
    except Exception:
        return default


def collect_business_outputs(cleaned: dict, feats: dict) -> dict:
    """Gather each page's structured output by CALLING the same analytics
    functions the pages use (operational_analytics) and reading the same
    persisted forecast outputs the forecast pages display. No page logic is
    recomputed here.

    Each page is collected independently and defensively: if a page is later
    removed, or its data/output is unavailable, that entry becomes None and the
    other pages are unaffected. Nothing is hardcoded — every value is read live
    from the current processed dataframes / persisted outputs."""
    inv = cleaned.get("invoices")
    pm = feats.get("property_month")

    def _current_occupancy():
        s = (pm["occupancy_pct"].dropna()
             if pm is not None and "occupancy_pct" in pm.columns
             else pd.Series(dtype=float))
        return {
            "current_pct": float(s.iloc[-1]) if len(s) else None,
            "forecast": _load_csv("forecast_occupancy_pct.csv"),
        }

    return {
        # Financial Overview page
        "financial": _safe(lambda: {
            "fy": ops.financial_year_revenue(inv),
            "by_fy": ops.revenue_by_financial_year(inv),
        }),
        # Executive Summary / Occupancy — booking-based occupancy (page's source)
        "occupancy": _safe(_current_occupancy),
        # Revenue Forecast page
        "revenue_forecast": _safe(
            lambda: _load_json("model_meta_multivariate.json")),
        # Electricity / Apartment-wise Forecast pages
        "electricity": _safe(lambda: {
            "summary": _load_csv("forecast_summary.csv"),
            "apartment": _load_csv("forecast_apartment_summary.csv"),
        }),
        # Notice & Exit page
        "notices": _safe(lambda: ops.notice_analytics(cleaned["notices"])),
        # Maintenance page
        "maintenance": _safe(lambda: ops.maintenance_summary(cleaned["tickets"])),
        # Available Beds page
        "beds": _safe(lambda: ops.bed_availability(cleaned["beds_snapshot"])),
        # Tenant Segmentation page
        "segments": _safe(lambda: _load_csv("tenant_segments_profile.csv")),
    }


# --------------------------------------------------------------------------- #
# Generate — summarise the collected outputs into prioritised recommendations
# --------------------------------------------------------------------------- #
def generate_recommendations(o: dict) -> list[dict]:
    """Turn the structured page outputs in `o` into prioritised recommendation
    cards. Reads ONLY from `o` — no dataset or model access here, so every value
    traces back to a dashboard page's output."""
    recs: list[dict] = []

    def add(priority, category, icon, source, recommendation, impact):
        recs.append(dict(priority=priority, category=category, icon=icon,
                         source=source, recommendation=recommendation,
                         impact=impact))

    # 📈 Revenue — Financial Overview (current month) vs Revenue Forecast.
    fin = (o.get("financial") or {}).get("fy")
    rf = o.get("revenue_forecast")
    monthly = fin.get("monthly") if fin else None
    if rf and rf.get("next_month_revenue") and monthly is not None and len(monthly):
        cur = float(monthly.iloc[-1]["revenue"])
        nxt = float(rf["next_month_revenue"])
        d = nxt - cur
        pct = d / cur * 100 if cur else 0
        if d >= 0:
            add("Low", "Revenue", "📈", "Financial Overview + Revenue Forecast",
                f"Revenue forecast to rise {pct:+.1f}% next month "
                f"(₹{cur/1e5:.1f}L → ₹{nxt/1e5:.1f}L). Hold occupancy and service "
                "quality to keep the trend.",
                f"+₹{d/1e5:.2f}L projected next month.")
        else:
            add("High", "Revenue", "📈", "Financial Overview + Revenue Forecast",
                f"Revenue forecast to fall {pct:.1f}% next month "
                f"(₹{cur/1e5:.1f}L → ₹{nxt/1e5:.1f}L). Refill vacant beds and lift "
                "occupancy to defend revenue.",
                f"₹{abs(d)/1e5:.2f}L of monthly revenue at stake.")

    # 🛏️ Occupancy — Executive Summary current vs Occupancy Forecast.
    occ = o.get("occupancy") or {}
    cur_occ = occ.get("current_pct")
    occ_fc = occ.get("forecast")
    if cur_occ is not None and occ_fc is not None and len(occ_fc):
        nxt_occ = float(occ_fc.iloc[0]["occupancy_pct"])
        nxt_beds = int(occ_fc.iloc[0]["occupied_beds"])
        vacant = config.TOTAL_BEDS - nxt_beds
        if nxt_occ < cur_occ - 0.5:
            add("High", "Occupancy", "🛏️", "Occupancy Forecast",
                f"Occupancy forecast to drop {cur_occ:.1f}% → {nxt_occ:.1f}% next "
                f"month. Market the {vacant} vacant beds now — push high-rate beds "
                "first.",
                f"~{vacant} beds to refill to hold occupancy.")
        elif nxt_occ >= 95:
            add("Medium", "Occupancy", "🛏️", "Occupancy Forecast",
                f"Occupancy forecast high at {nxt_occ:.1f}% "
                f"({nxt_beds}/{config.TOTAL_BEDS} beds). Prepare staffing, "
                "maintenance and onboarding for near-full capacity.",
                "Protect service quality at peak occupancy.")
        else:
            add("Low", "Occupancy", "🛏️", "Occupancy Forecast",
                f"Occupancy stable ({cur_occ:.1f}% → {nxt_occ:.1f}%). Maintain "
                "current retention and intake pace.",
                "Stable occupancy supports the revenue forecast.")

    # ⚡ Electricity cost outlook — Electricity forecast page.
    es = (o.get("electricity") or {}).get("summary")
    if es is not None and "series" in es.columns:
        el = es[es["series"] == "elec_cost"]
        if len(el):
            add("Low", "Electricity", "⚡", "Electricity Forecast",
                f"Next-month electricity cost forecast "
                f"₹{float(el['next_month'].iloc[0])/1e5:.1f}L "
                f"(MAPE {float(el['MAPE'].iloc[0]):.1f}%). Budget accordingly.",
                "Accurate operating-cost budgeting.")

    # 🚪 Exit notices — Notice & Exit page.
    nt = o.get("notices") or {}
    ue = int(nt.get("upcoming_exits", 0))
    if ue:
        add("High" if ue >= 5 else "Medium", "Exit Notices", "📤", "Notice & Exit",
            f"{ue} tenants scheduled to vacate. Start replacement bookings now and "
            "begin retention outreach for notice beds.",
            f"₹{float(nt.get('monthly_revenue_impact', 0))/1e5:.2f}L monthly rent "
            "at stake across notices.")

    # 🏠 Available beds — Available Beds page.
    bd = o.get("beds") or {}
    vac = int(bd.get("vacant_beds", 0))
    if vac and len(bd.get("by_block", [])):
        blk = bd["by_block"].iloc[0]
        add("Medium", "Available Beds", "🏠", "Available Beds",
            f"{vac} beds vacant. Promote the highest-vacancy block "
            f"'{blk['block']}' ({blk['vacancy_pct']:.0f}% vacant) and re-list "
            "high-rate beds first.",
            f"₹{float(bd.get('vacant_revenue_opportunity', 0))/1e5:.2f}L/mo revenue "
            "opportunity.")

    # 🔧 Maintenance — Maintenance page.
    ms = o.get("maintenance") or {}
    if ms.get("open"):
        sla = ms.get("sla_breach_pct")
        top_issue = (ms["by_issue"].iloc[0]["issue_type"]
                     if len(ms.get("by_issue", [])) else "open tickets")
        add("Medium", "Maintenance", "🔧", "Maintenance",
            f"{ms['open']} open maintenance tickets"
            + (f", SLA breached on {sla}%" if sla else "")
            + f". Clear the backlog — top issue: {top_issue}.",
            "Faster resolution lifts tenant retention.")

    # 👥 Tenant segments — Tenant Segmentation page.
    seg = o.get("segments")
    if seg is not None and len(seg) and "ltv_paid" in seg.columns:
        top = seg.sort_values("ltv_paid", ascending=False).iloc[0]
        add("Low", "Tenant Segments", "👥", "Tenant Segmentation",
            f"{int(top['n_tenants'])} tenants in the top-value segment average "
            f"₹{float(top['ltv_paid'])/1e5:.2f}L lifetime value. Protect them with "
            "priority service and renewal offers.",
            "Retain the highest-value tenants.")

    recs.sort(key=lambda c: _ORDER.get(c["priority"], 3))
    return recs


def recommend(cleaned: dict, feats: dict) -> tuple[dict, list[dict]]:
    """Convenience: collect page outputs then summarise them. Returns
    (business_outputs, recommendations)."""
    outputs = collect_business_outputs(cleaned, feats)
    return outputs, generate_recommendations(outputs)


if __name__ == "__main__":
    from src import preprocessing
    from src import feature_engineering as fe
    cl = preprocessing.clean_all()
    ft = fe.build_all(cl)
    outs, recs = recommend(cl, ft)
    print(f"pages collected: {list(outs)}")
    print(f"recommendations: {len(recs)}")
    for r in recs:
        print(f"  [{r['priority']:6}] {r['category']:16} <- {r['source']}")
