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

# Pricing Opportunity — data-driven demand thresholds (no apartment/rent hardcoding).
# Live: operational occupancy at/above this AND zero vacant beds.
# History: mean monthly occupancy over the recent window at/above this.
# Rent: current avg rent at/below the portfolio median (headroom to raise).
PRICING_LIVE_OCC_PCT = 95.0
PRICING_HIST_OCC_PCT = 90.0
PRICING_LOOKBACK_MONTHS = 6
PRICING_MAX_CARDS = 1


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

    def _pricing_opportunity():
        """Apartments with consistently high demand and low vacancy — candidates
        for a moderate rent increase on future bookings. Built live from
        live_bed_snapshot (current operational occupancy/vacancy/rent) + booking
        history (recent monthly occupancy). No hardcoded apartments or rents.
        Returns empty when nothing qualifies (caller hides the card)."""
        if snap is None or not len(snap) or bookings is None or not len(bookings):
            return None
        tenant_states = ["Occupied", "Notice", "Notice-Booked"]
        apt = (snap.groupby("apartment_code", as_index=False)
                   .agg(total=("bed_id", "count"),
                        inactive=("is_inactive", "sum"),
                        occupied_now=("live_status",
                                      lambda s: int(s.isin(tenant_states).sum())),
                        vacant=("is_vacant", "sum"),
                        avg_rent=("current_rate", "mean"),
                        booking_activity=("live_status",
                                          lambda s: int(s.isin(
                                              tenant_states + ["Booked"]).sum()))))
        apt["operational"] = apt["total"] - apt["inactive"]
        apt = apt[apt["operational"] > 0].copy()
        if not len(apt):
            return pd.DataFrame()
        apt["occ_pct"] = (apt["occupied_now"] / apt["operational"] * 100).round(1)

        # Portfolio rent median — data-driven pricing headroom benchmark.
        rent_med = float(apt["avg_rent"].dropna().median()) \
            if apt["avg_rent"].notna().any() else float("nan")

        # Recent monthly occupancy from booking stay intervals (real data only).
        b = bookings.dropna(subset=["bed_code", "apartment_code"]).copy()
        b["_bed"] = (b["apartment_code"].astype("string") + "|"
                     + b["bed_code"].astype("string"))
        onboard = pd.to_datetime(b["onboarding_date"], utc=True, errors="coerce")
        exit_dt = pd.to_datetime(b["actual_exit_date"], utc=True, errors="coerce")
        now = pd.Timestamp.now(tz="UTC")
        months = pd.period_range(end=now.to_period("M"),
                                 periods=PRICING_LOOKBACK_MONTHS, freq="M")
        op_map = apt.set_index("apartment_code")["operational"]
        hist_rows = []
        for m in months:
            s, e = m.start_time.tz_localize("UTC"), m.end_time.tz_localize("UTC")
            live = b[(onboard <= e) & (exit_dt.isna() | (exit_dt >= s))]
            occ_n = live.groupby("apartment_code")["_bed"].nunique()
            for a, op in op_map.items():
                hist_rows.append({
                    "apartment_code": a,
                    "month": str(m),
                    "occ_pct": (float(occ_n.get(a, 0)) / op * 100) if op else 0.0,
                })
        hist = pd.DataFrame(hist_rows)
        hist_avg = (hist.groupby("apartment_code")["occ_pct"].mean()
                        .rename("hist_occ_pct"))
        full_months = (hist.assign(full=(hist["occ_pct"] >= PRICING_LIVE_OCC_PCT))
                           .groupby("apartment_code")["full"].mean()
                           .rename("full_month_share"))

        out = apt.merge(hist_avg, on="apartment_code", how="left")
        out = out.merge(full_months, on="apartment_code", how="left")
        out["hist_occ_pct"] = out["hist_occ_pct"].fillna(0.0).round(1)
        out["full_month_share"] = out["full_month_share"].fillna(0.0)

        # High live demand + consistently low vacancy + rent at/below median.
        hit = out[
            (out["occ_pct"] >= PRICING_LIVE_OCC_PCT)
            & (out["vacant"] == 0)
            & (out["hist_occ_pct"] >= PRICING_HIST_OCC_PCT)
            & (out["avg_rent"].notna())
            & (out["avg_rent"] <= rent_med if rent_med == rent_med else True)
        ].copy()
        if not len(hit):
            return pd.DataFrame()
        hit["portfolio_median_rent"] = (
            round(rent_med, 0) if rent_med == rent_med else None)
        # Rank strongest demand first: sustained high occupancy, zero vacancy,
        # then business impact (more filled beds + more rent headroom vs median).
        hit["rent_headroom"] = (
            (rent_med - hit["avg_rent"]) if rent_med == rent_med else 0.0)
        hit = hit.sort_values(
            ["hist_occ_pct", "full_month_share", "occ_pct", "vacant",
             "operational", "rent_headroom"],
            ascending=[False, False, False, True, False, False]
        ).head(PRICING_MAX_CARDS)
        return hit[["apartment_code", "occ_pct", "hist_occ_pct", "vacant",
                    "operational", "occupied_now", "avg_rent",
                    "portfolio_median_rent", "booking_activity",
                    "full_month_share"]].reset_index(drop=True)

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
        "pricing_opportunity": _safe(_pricing_opportunity),
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
                f"Latest monthly revenue is ₹{cur/1e5:.1f}L "
                f"({pct:+.1f}% vs the prior month). "
                "Consider monitoring occupancy and collections to sustain the trend.",
                f"Month-on-month change: +₹{d/1e5:.2f}L.")
        else:
            add("High", "Revenue", "📈", "property_month",
                f"Latest monthly revenue is ₹{cur/1e5:.1f}L "
                f"({pct:.1f}% vs the prior month). "
                "Consider reviewing vacancy and collection performance for the "
                "current period.",
                f"Month-on-month change: −₹{abs(d)/1e5:.2f}L.")

    # 🛏️ Occupancy — property_month + occupancy forecast.
    if pm is not None and len(pm) and "occupancy_pct" in pm.columns:
        p = pm.sort_values("billing_period")
        cur_occ = float(p["occupancy_pct"].iloc[-1])
        ofc = o.get("occupancy_forecast")
        nxt = (float(ofc.iloc[0]["occupancy_pct"])
               if ofc is not None and len(ofc) else None)
        if nxt is not None and nxt < cur_occ - 0.5:
            add("High", "Occupancy", "🛏️", "property_month + Occupancy Forecast",
                f"Occupancy is {cur_occ:.1f}% currently, with next-month forecast "
                f"at {nxt:.1f}%. Consider reviewing available inventory and intake "
                "pipeline for the coming period.",
                "Forecast suggests lower occupancy vs the current level.")
        elif cur_occ >= 95:
            add("Low", "Occupancy", "🛏️", "property_month",
                f"Occupancy is {cur_occ:.1f}%. "
                "Consider continuing to monitor retention and intake to maintain "
                "current levels.",
                "Occupancy is currently in a strong range.")
        else:
            add("Medium", "Occupancy", "🛏️", "property_month",
                f"Occupancy is {cur_occ:.1f}%. "
                "Consider reviewing vacant inventory and leasing activity for "
                "improvement opportunities.",
                "Occupancy has room to improve relative to capacity.")

    # 💵 Pricing Opportunity — high-demand / low-vacancy apartments (dynamic).
    # Shown ONLY when at least one apartment qualifies; never a blank/all-clear card.
    # Exactly ONE card: the single highest-ranked apartment (ranking unchanged).
    po = o.get("pricing_opportunity")
    if po is not None and len(po):
        r = po.iloc[0]   # top-ranked only — never emit multiple Pricing cards
        rent = float(r["avg_rent"])
        med = r.get("portfolio_median_rent")
        med_txt = (f" (portfolio median ₹{int(med):,})"
                   if med is not None and med == med else "")
        full_share = float(r.get("full_month_share", 0.0)) * 100
        add("Medium", "Pricing Opportunity", "💵",
            "live_bed_snapshot + booking history",
            f"Apartment {r['apartment_code']} is at {r['occ_pct']:.1f}% "
            f"operational occupancy with {int(r['vacant'])} vacant beds "
            f"({int(r['occupied_now'])}/{int(r['operational'])} beds filled). "
            f"Recent {PRICING_LOOKBACK_MONTHS}-month average occupancy is "
            f"{r['hist_occ_pct']:.1f}% ({full_share:.0f}% of months at/above "
            f"{PRICING_LIVE_OCC_PCT:.0f}%). Current average rent ₹{rent:,.0f}"
            f"{med_txt}. Consider a rent review for future bookings while "
            "monitoring occupancy trends.",
            "Sustained demand with low vacancy may support a measured pricing "
            "review on new leases.")

    # 🏠 Available Beds — live bed snapshot (SAME logic as the Available Beds page).
    vac = o.get("vacant_beds_operational")
    if vac is not None:
        if vac:
            top = o.get("top_vacant_apartment")
            detail = (f" Highest vacancy currently: {top[0]} ({top[1]} beds)."
                      if top else "")
            opp = float(o.get("vacant_revenue_opportunity", 0.0))
            add("Medium", "Available Beds", "🏠", "live_bed_snapshot",
                f"{vac} operational beds are currently vacant.{detail} "
                "Consider prioritising leasing activity on higher-rate vacant beds.",
                f"Estimated monthly revenue opportunity: ₹{opp/1e5:.2f}L.")
        else:
            add("Low", "Available Beds", "🏠", "live_bed_snapshot",
                "No operational vacant beds at present.",
                "Operational inventory is currently fully occupied.")

    # 📤 Notice & Exit — active notices only.
    na = o.get("notices_active")
    if na is not None:
        n = len(na)
        if n:
            risk = (float(na["monthly_rental"].sum())
                    if "monthly_rental" in na.columns else 0.0)
            add("High" if n >= 5 else "Medium", "Notice & Exit", "📤",
                "notices (active)",
                f"{n} active exit notices at present. "
                "Consider reviewing replacement pipeline and retention options "
                "for affected beds.",
                f"Associated monthly rent at risk: ₹{risk/1e5:.2f}L.")
        else:
            add("Low", "Notice & Exit", "📤", "notices (active)",
                "No active exit notices at present.",
                "No imminent exit-related churn indicated in the current data.")

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
                + (f"; SLA breached on {sla}%" if sla else "")
                + f". Leading issue type: {top}. "
                "Consider prioritising resolution of the current backlog.",
                "Timely maintenance support may help sustain tenant satisfaction.")
        else:
            add("Low", "Maintenance", "🔧", "tickets",
                "No open maintenance tickets at present.",
                "Maintenance backlog is currently clear.")

    # 🔮 Forecast — forecasting outputs (auto-selected model).
    fs = o.get("forecast_summary")
    sel = o.get("revenue_forecast_selected")
    if fs is not None and len(fs[fs["series"] == "revenue"]):
        r = fs[fs["series"] == "revenue"].iloc[0]
        model = sel["winner"] if sel else r["method"]
        add("Low", "Forecast", "🔮", "Revenue Forecast",
            f"Next-month revenue forecast is ₹{float(r['next_month'])/1e5:.1f}L "
            f"(model: {model}, walk-forward MAPE {float(r['MAPE']):.1f}%). "
            "Consider using this outlook for near-term planning.",
            "Provides forward visibility based on the current forecast model.")

    # ⚡ Electricity alert — apartment_features (vacant apartment drawing power).
    ea = o.get("electricity_alert")
    if ea is not None and len(ea):
        for _, r in ea.iterrows():
            units = int(round(float(r["units_consumed"])))
            add("Medium", "Electricity", "⚡", "apartment_features",
                f"Apartment {r['apartment_code']} shows vacant status with "
                f"{units} units consumed in the latest reading. "
                "Consider reviewing the meter reading before the next billing cycle.",
                "Possible electricity usage on a vacant apartment.")

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
                f"Apartment {r['apartment_code']} was flagged by the IsolationForest "
                f"electricity detector — {units} units"
                f"{(' in ' + bp) if bp else ''}{score_txt}. "
                "Possible causes include a meter or billing outlier, unusually high "
                "consumption, or a data issue. Consider reviewing the reading before "
                "the next cycle.",
                "ML-detected electricity outlier for follow-up review.")

    if has_vacant:
        for _, r in en.iterrows():
            add("High", "Vacant Room Energy Alert", "⚡",
                "live_bed_snapshot + electricity",
                f"Apartment {r['apartment_code']} is currently {r['status']} but "
                f"consumed {int(r['units_consumed'])} units in "
                f"{r['billing_period']}. "
                "Inspect the room for electrical leakage, appliances left running, "
                "or meter issues.",
                "Reduce unnecessary electricity costs in unoccupied rooms.")

    if (ml is not None or en is not None) and not has_ml and not has_vacant:
        add("Low", "Energy anomaly detected", "⚡",
            "IsolationForest + vacant-room rule",
            "No electricity anomalies flagged by the IsolationForest detector or "
            "the vacant-room leakage check in the current data.",
            "No electricity anomalies indicated by either check at present.")

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
