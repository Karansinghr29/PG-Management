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
from src import forecasting as fc  # noqa: E402
from src import bed_snapshot as bs  # noqa: E402
from src import revenue_multivariate as rmv  # noqa: E402
from src import anomaly as anom  # noqa: E402
from src import feature_engineering as fe  # noqa: E402

_ORDER = {"High": 0, "Medium": 1, "Low": 2}

# Energy-anomaly threshold as a fraction of the occupied-room monthly-units
# median (data-driven, not a hardcoded kWh figure): a room with no paying tenant
# drawing at least this share of a normal occupied room's usage is flagged.
ENERGY_ANOMALY_FRACTION = 0.15


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
    """Gather the LATEST processed dataframes + forecasting outputs, live on every
    call (never cached). Uses ONLY: property_month, apartment_features,
    tenant_features, bed_features, notices, tickets, the live bed snapshot and a
    live in-memory forecast — no persisted CSV/JSON, no raw-dataset calculations.
    Each source is collected defensively; a missing source becomes None and the
    rest are unaffected."""
    pm = feats.get("property_month")
    af = feats.get("apartment_features")
    tf = feats.get("tenant_features")
    bf = feats.get("bed_features")
    notices = cleaned.get("notices")
    tickets = cleaned.get("tickets")
    bookings = cleaned.get("bookings")
    electricity = cleaned.get("electricity")

    def _active_notices():
        if notices is None or not len(notices):
            return notices
        ex = pd.to_datetime(notices["estimated_exit_date"], utc=True,
                            errors="coerce")
        return notices[ex >= pd.Timestamp.now(tz="UTC")]

    def _elec_alert():
        # Vacant apartment drawing power — derived from apartment_features only.
        if af is None or not len(af) or "occupancy_pct" not in af.columns:
            return None
        hit = af[(af["occupancy_pct"] == 0) & (af.get("elec_units", 0) > 0)]
        return pd.DataFrame({"apartment_code": hit["apartment_code"].to_numpy(),
                             "units_consumed": hit["elec_units"].to_numpy()})

    # Live operational bed snapshot — SAME source of truth as the Available Beds
    # page (src/bed_snapshot.py), so AI vacancy matches that page exactly.
    snap = _safe(lambda: bs.live_bed_snapshot(bookings))

    def _energy_anomaly():
        """Vacant/Inactive rooms drawing electricity — live_bed_snapshot (current
        room status) x electricity (latest-month units). A room with NO current
        occupants should draw ~0; flag those consuming at least
        ENERGY_ANOMALY_FRACTION of the occupied-room median (data-driven). Real
        data only — no synthetic values."""
        if (snap is None or not len(snap)
                or electricity is None or not len(electricity)):
            return None
        e = electricity.copy()
        e["_bp"] = e["billing_period"].astype(str)
        latest = e.sort_values("_bp").groupby("apartment_code").tail(1)
        period = latest["_bp"].max()
        units = latest.set_index("apartment_code")["units_consumed"]
        tenant_states = ["Occupied", "Notice", "Notice-Booked"]
        apt = (snap.groupby("apartment_code")
                   .agg(total=("bed_id", "count"),
                        inactive=("is_inactive", "sum"),
                        occupied_now=("live_status",
                                      lambda s: int(s.isin(tenant_states).sum()))))
        apt["operational"] = apt["total"] - apt["inactive"]
        occ_codes = [a for a in apt.index[apt["occupied_now"] > 0]
                     if a in units.index]
        occ_units = units.loc[occ_codes].dropna()
        baseline = float(occ_units.median()) if len(occ_units) else float("nan")
        thr = ENERGY_ANOMALY_FRACTION * baseline if baseline == baseline else 0.0
        rows = []
        for a in apt.index[apt["occupied_now"] == 0]:
            u = float(units.get(a, float("nan")))
            if u == u and u > 0 and (thr <= 0 or u >= thr):
                rows.append({
                    "apartment_code": a,
                    "units_consumed": round(u, 0),
                    "billing_period": period,
                    "status": ("Inactive" if apt.loc[a, "operational"] == 0
                               else "Vacant"),
                    "baseline_units": (round(baseline, 0)
                                       if baseline == baseline else None),
                    "threshold_units": round(thr, 0)})
        return (pd.DataFrame(rows).sort_values("units_consumed", ascending=False)
                if rows else pd.DataFrame())

    def _ml_elec_anomaly():
        """IsolationForest electricity anomalies — same detector as src/anomaly.py
        (units_consumed, amount, deviation_from_avg). Model unchanged; this only
        exposes its flagged rows to AI Recommendations. Uses electricity_features
        so deviation_from_avg is present on the live cleaned electricity table."""
        if electricity is None or not len(electricity):
            return None
        efeat = feats.get("electricity_features")
        if efeat is None or not len(efeat) or "deviation_from_avg" not in efeat.columns:
            efeat = fe.electricity_features(electricity)
        cols = ["units_consumed", "amount", "deviation_from_avg"]
        if not set(cols).issubset(efeat.columns):
            return pd.DataFrame()
        scored = anom._detect(efeat, cols)
        hit = scored[scored["anomaly"] == 1].sort_values(
            "anomaly_score", ascending=False)
        keep = [c for c in ["apartment_code", "billing_period", "units_consumed",
                            "amount", "anomaly_score", "deviation_from_avg"]
                if c in hit.columns]
        return hit[keep].reset_index(drop=True) if len(hit) else pd.DataFrame()

    vacant_operational = None
    vacant_opportunity = 0.0
    top_vacant = None
    if snap is not None and len(snap):
        vac = snap[snap["live_status"] == "Vacant"]
        vacant_operational = int(len(vac))
        vacant_opportunity = float(vac["current_rate"].fillna(0).sum())
        if vacant_operational:
            by_apt = (vac.groupby("apartment_code").size()
                         .sort_values(ascending=False))
            top_vacant = (str(by_apt.index[0]), int(by_apt.iloc[0]))

    # Revenue forecast — SINGLE SOURCE OF TRUTH: the live Ridge multivariate model
    # (same value shown on the Revenue Forecast KPI card + Financial Overview).
    # Occupancy forecast stays on the live Holt-Winters time series (kept for that
    # purpose). Both are live in-memory: no file writes, no stale JSON/CSV.
    mv_pred = _safe(lambda: rmv.predict_live(pm)) if pm is not None else None
    occ = _safe(lambda: fc.forecast_live(pm, series=("occupied_beds",))) \
        if pm is not None else None
    occ = occ or {}
    fsummary = None
    if mv_pred is not None:
        fsummary = pd.DataFrame([{
            "series": "revenue", "method": f"{mv_pred['model']}_multivariate",
            "MAE": mv_pred["mae"], "RMSE": mv_pred["rmse"],
            "MAPE": mv_pred["mape"], "windows": None,
            "next_month": mv_pred["next_month_revenue"]}])

    return {
        "property_month": _safe(lambda: pm),
        "apartment_features": _safe(lambda: af),
        "tenant_features": _safe(lambda: tf),
        "bed_features": _safe(lambda: bf),
        "notices_active": _safe(_active_notices),
        "maintenance": _safe(lambda: ops.maintenance_summary(tickets)),
        "occupancy_forecast": occ.get("occupancy_forecast"),
        "forecast_summary": fsummary,
        "revenue_forecast_selected": None,   # model label falls back to method
        "electricity_alert": _safe(_elec_alert),
        "energy_anomaly": _safe(_energy_anomaly),
        "ml_elec_anomaly": _safe(_ml_elec_anomaly),
        "bed_snapshot": snap,
        "vacant_beds_operational": vacant_operational,
        "vacant_revenue_opportunity": vacant_opportunity,
        "top_vacant_apartment": top_vacant,
    }


# --------------------------------------------------------------------------- #
# Generate — summarise the collected outputs into prioritised recommendations
# --------------------------------------------------------------------------- #
def generate_recommendations(o: dict) -> list[dict]:
    """Turn the collected processed outputs in `o` into prioritised recommendation
    cards. Reads ONLY from `o`. Each category always emits a card — an action when
    there is an issue, a positive note when there is none (never blank)."""
    recs: list[dict] = []

    def add(priority, category, icon, source, recommendation, impact):
        recs.append(dict(priority=priority, category=category, icon=icon,
                         source=source, recommendation=recommendation,
                         impact=impact))

    # 📈 Revenue — property_month (latest vs previous month).
    pm = o.get("property_month")
    if pm is not None and len(pm):
        p = pm.sort_values("billing_period")
        cur = float(p["revenue"].iloc[-1])
        prev = float(p["revenue"].iloc[-2]) if len(p) >= 2 else cur
        d = cur - prev
        pct = (d / prev * 100) if prev else 0
        if d >= 0:
            add("Low", "Revenue", "📈", "property_month",
                f"Monthly revenue up {pct:+.1f}% to ₹{cur/1e5:.1f}L. Sustain "
                "occupancy and collections to hold the trend.",
                f"Revenue trending up (+₹{d/1e5:.2f}L MoM).")
        else:
            add("High", "Revenue", "📈", "property_month",
                f"Monthly revenue down {pct:.1f}% to ₹{cur/1e5:.1f}L. Refill vacant "
                "beds and accelerate collections to defend revenue.",
                f"₹{abs(d)/1e5:.2f}L revenue drop MoM.")

    # 🛏️ Occupancy — property_month + occupancy forecast.
    if pm is not None and len(pm) and "occupancy_pct" in pm.columns:
        p = pm.sort_values("billing_period")
        cur_occ = float(p["occupancy_pct"].iloc[-1])
        ofc = o.get("occupancy_forecast")
        nxt = (float(ofc.iloc[0]["occupancy_pct"])
               if ofc is not None and len(ofc) else None)
        if nxt is not None and nxt < cur_occ - 0.5:
            add("High", "Occupancy", "🛏️", "property_month + Occupancy Forecast",
                f"Occupancy forecast to drop {cur_occ:.1f}% → {nxt:.1f}%. Market the "
                "vacant beds now.", "Falling occupancy risks next-month revenue.")
        elif cur_occ >= 95:
            add("Low", "Occupancy", "🛏️", "property_month",
                f"Occupancy strong at {cur_occ:.1f}%. Maintain retention and intake "
                "pace.", "High occupancy supports revenue.")
        else:
            add("Medium", "Occupancy", "🛏️", "property_month",
                f"Occupancy at {cur_occ:.1f}%. Fill vacant beds to lift it.",
                "Room to grow occupancy.")

    # 🏠 Available Beds — live bed snapshot (SAME logic as the Available Beds page).
    vac = o.get("vacant_beds_operational")
    if vac is not None:
        if vac:
            top = o.get("top_vacant_apartment")
            detail = (f" Highest-vacancy apartment {top[0]} ({top[1]} beds)."
                      if top else "")
            opp = float(o.get("vacant_revenue_opportunity", 0.0))
            add("Medium", "Available Beds", "🏠", "live_bed_snapshot",
                f"{vac} beds vacant.{detail} Re-list high-rate vacant beds first.",
                f"₹{opp/1e5:.2f}L/mo revenue opportunity.")
        else:
            add("Low", "Available Beds", "🏠", "live_bed_snapshot",
                "No vacant beds — the property is fully occupied.",
                "Occupancy maximised.")

    # 📤 Notice & Exit — active notices only.
    na = o.get("notices_active")
    if na is not None:
        n = len(na)
        if n:
            risk = (float(na["monthly_rental"].sum())
                    if "monthly_rental" in na.columns else 0.0)
            add("High" if n >= 5 else "Medium", "Notice & Exit", "📤",
                "notices (active)",
                f"{n} active exit notices. Start replacement bookings and begin "
                "retention outreach.",
                f"₹{risk/1e5:.2f}L monthly rent at risk.")
        else:
            add("Low", "Notice & Exit", "📤", "notices (active)",
                "No active exit notices — the tenant base is stable.",
                "No imminent churn.")

    # 🔧 Maintenance — tickets.
    ms = o.get("maintenance") or {}
    if ms:
        opn = int(ms.get("open", 0))
        if opn:
            sla = ms.get("sla_breach_pct")
            top = (ms["by_issue"].iloc[0]["issue_type"]
                   if len(ms.get("by_issue", [])) else "open tickets")
            add("Medium", "Maintenance", "🔧", "tickets",
                f"{opn} open maintenance tickets"
                + (f", SLA breached on {sla}%" if sla else "")
                + f". Clear the backlog — top issue: {top}.",
                "Faster resolution lifts tenant retention.")
        else:
            add("Low", "Maintenance", "🔧", "tickets",
                "No open maintenance tickets — all resolved.",
                "Maintenance under control.")

    # 👥 Tenant Segmentation — tenant_features.
    tf = o.get("tenant_features")
    if tf is not None and len(tf) and "ltv_paid" in tf.columns:
        add("Low", "Tenant Segmentation", "👥", "tenant_features",
            f"{len(tf):,} tenants; average lifetime value "
            f"₹{tf['ltv_paid'].mean()/1e5:.2f}L. Protect high-LTV tenants with "
            "priority service and renewals.",
            "Retain the highest-value tenants.")

    # 🔮 Forecast — forecasting outputs (auto-selected model).
    fs = o.get("forecast_summary")
    sel = o.get("revenue_forecast_selected")
    if fs is not None and len(fs[fs["series"] == "revenue"]):
        r = fs[fs["series"] == "revenue"].iloc[0]
        model = sel["winner"] if sel else r["method"]
        add("Low", "Forecast", "🔮", "Revenue Forecast",
            f"Next-month revenue forecast ₹{float(r['next_month'])/1e5:.1f}L "
            f"(model: {model}, walk-forward MAPE {float(r['MAPE']):.1f}%). Use for "
            "cash-flow planning.", "Forward visibility for planning.")

    # ⚡ Electricity alert — apartment_features (vacant apartment drawing power).
    ea = o.get("electricity_alert")
    if ea is not None and len(ea):
        for _, r in ea.iterrows():
            units = int(round(float(r["units_consumed"])))
            add("Medium", "Electricity", "⚡", "apartment_features",
                f"Apartment {r['apartment_code']} is vacant but consumed {units} "
                "units. Inspect the meter before the next billing cycle.",
                "Electricity leakage in a vacant apartment.")

    # ⚡ Energy anomalies — two complementary checks (neither changes the other):
    #   1) IsolationForest ML electricity anomalies (src/anomaly._detect)
    #   2) Vacant/inactive room electricity leakage (existing energy_anomaly rule)
    # "No energy anomalies" only when BOTH return empty.
    ml = o.get("ml_elec_anomaly")
    en = o.get("energy_anomaly")
    has_ml = ml is not None and len(ml) > 0
    has_vacant = en is not None and len(en) > 0

    if has_ml:
        for _, r in ml.iterrows():
            units = int(round(float(r["units_consumed"])))
            bp = str(r["billing_period"]) if "billing_period" in r.index else ""
            score = float(r["anomaly_score"]) if "anomaly_score" in r.index else None
            score_txt = f", anomaly score {score:.2f}" if score is not None else ""
            add("High", "ML electricity anomaly detected", "⚡",
                "IsolationForest (anomaly.py)",
                f"Apartment {r['apartment_code']} flagged by the IsolationForest "
                f"electricity detector — {units} units"
                f"{(' in ' + bp) if bp else ''}{score_txt}. "
                "Possible causes: meter/billing outlier, unusually high consumption, "
                "or a data error. Review the reading before the next cycle.",
                "ML-detected electricity outlier — investigate meter and billing.")

    if has_vacant:
        for _, r in en.iterrows():
            base = (f"~{int(r['baseline_units'])}-unit occupied-room median"
                    if r.get("baseline_units") is not None else "occupied-room usage")
            add("High", "Vacant room consuming abnormal electricity", "⚡",
                "live_bed_snapshot + electricity",
                f"Room {r['apartment_code']} is {r['status']} (no current tenant) "
                f"but consumed {int(r['units_consumed'])} units in "
                f"{r['billing_period']}, above the data-driven threshold of "
                f"{int(r['threshold_units'])} units ({base}). "
                "Possible causes: AC / geyser / lights left running, a meter "
                "mis-mapping, unauthorised usage, or a faulty meter. Inspect the "
                "room and its meter before the next billing cycle.",
                "Electricity cost leakage on a room earning no rent — direct "
                "margin loss and a possible metering/billing error.")

    if (ml is not None or en is not None) and not has_ml and not has_vacant:
        add("Low", "Energy anomaly detected", "⚡",
            "IsolationForest + vacant-room rule",
            "No energy anomalies — neither the IsolationForest electricity detector "
            "nor the vacant-room leakage rule flagged any apartment.",
            "No electricity anomalies detected by either check.")

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
