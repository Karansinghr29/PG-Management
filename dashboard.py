""" PG Management Analytics dashboard.

Production-style Streamlit + Plotly application over the real PG datasets.
Every number shown is computed from actual data or persisted model outputs -
no synthetic data, no fake joins.

    streamlit run dashboard.py     # full interactive app
    python dashboard.py            # writes outputs/dashboard.html (static fallback)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import operational_analytics as ops  # noqa: E402
from src import recommendation_engine as rec  # noqa: E402
from src import preprocessing  # noqa: E402
from src import bed_snapshot as bs  # noqa: E402
from src import revenue_multivariate as rmv  # noqa: E402

APP_TITLE = "PG Management Analytics"

# Consistent, professional palette.
C_PRIMARY = "#2A9D8F"
C_ACCENT = "#264653"
C_WARN = "#E9C46A"
C_RISK = "#E76F51"
C_HIGH = "#C0392B"
C_MED = "#E67E22"
C_LOW = "#27AE60"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "vista_heights_chart",
                             "scale": 2},
}

PROBLEM_LABELS = {
    "late_payment": "Late Payment (classification)",
    "monthly_revenue": "Monthly Revenue (regression)",
    "electricity_cost": "Electricity Cost (regression)",
}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _data():
    """Reload ALL source datasets from disk and rebuild features, fresh, on every
    call. Intentionally NOT cached (no @st.cache_*): Streamlit reruns the whole
    script on each interaction, so this re-reads the raw CSVs and re-derives the
    feature tables every rerun. Business Insights and AI Recommendations are both
    generated from the objects returned here, so they always reflect the latest
    data and never reuse a previous run's dataframes."""
    cleaned = preprocessing.clean_all()   # reads the raw CSVs fresh from disk
    feats = fe.build_all(cleaned)          # rebuilds feature tables fresh
    return cleaned, feats


def _load_csv(name: str):
    p = config.OUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _load_meta(problem: str) -> dict | None:
    p = config.OUT_DIR / f"model_meta_{problem}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _kpi_cards(st, items: list[tuple[str, str]]):
    """Render a row of KPI cards from (label, value) pairs."""
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.markdown(f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
                   f"<div class='kpi-value'>{value}</div></div>",
                   unsafe_allow_html=True)


def _load_meta_json(filename: str) -> dict | None:
    p = config.OUT_DIR / filename
    if p.exists():
        return json.loads(p.read_text())
    return None


# Live bed classification now lives in a shared module so the Available Beds
# page and the AI Recommendations engine use the SAME source of truth.
_load_beds_inventory_table = bs.load_beds_inventory_table
_live_bed_snapshot = bs.live_bed_snapshot


def _live_bed_kpis_from_snapshot(snap: pd.DataFrame) -> dict:
    """KPIs from live_bed_snapshot. Vacant = Operational − Occupied − Notice − Notice-Booked − Booked."""
    physical = int(snap.attrs.get("physical_beds", len(snap)))
    operational = int(snap.attrs.get("operational_beds", physical))
    counts = snap["live_status"].value_counts() if len(snap) else pd.Series(dtype=int)
    occupied_beds = int(counts.get("Occupied", 0))
    notice_beds = int(counts.get("Notice", 0))
    notice_booked_beds = int(counts.get("Notice-Booked", 0))
    booked_beds = int(counts.get("Booked", 0))
    inactive_beds = int(counts.get("Inactive", 0))
    vacant = (operational - occupied_beds - notice_beds
              - notice_booked_beds - booked_beds)
    occupied_now = occupied_beds + notice_beds + notice_booked_beds
    occ_pct = (occupied_now / operational * 100) if operational else 0.0
    return dict(total_beds=physical, operational_beds=operational,
                occupied_beds=occupied_beds, notice_beds=notice_beds,
                notice_booked_beds=notice_booked_beds, booked_beds=booked_beds,
                inactive_beds=inactive_beds, vacant=vacant, occ_pct=occ_pct)


def _current_occupancy_pct(feats) -> float:
    """Single source of truth for CURRENT occupancy: booking-based occupancy%
    from property_month (occupied beds / TOTAL_BEDS), same series the Occupancy
    Forecast tab and the forecast are built on. Falls back to the bed snapshot
    only if the booking-based series is unavailable."""
    pm = feats["property_month"]
    if "occupancy_pct" in pm.columns and pm["occupancy_pct"].notna().any():
        return float(pm["occupancy_pct"].dropna().iloc[-1])
    beds = feats["bed_features"]
    return float(beds["is_occupied"].mean() * 100)


def _render_multivariate(st, feats):
    """Revenue+Occupancy multivariate block: comparison, correlation, importance,
    scenario, actual-vs-predicted. All from persisted real model outputs."""
    pm = feats["property_month"]
    if pm is None or not len(pm):
        st.info("No property_month data available for the revenue forecast.")
        return
    # Single source of truth: live Ridge multivariate prediction (same value the
    # AI Recommendations and Financial Overview use). Not read from the persisted
    # model_meta_multivariate.json.
    mv = rmv.predict_live(pm)
    mo = {"model": mv["model"], "mape": mv["mape"], "mae": mv["mae"]}
    mae = mo["mae"]                                   # for the 95% band
    next_rev = mv["next_month_revenue"]

    # ---- Headline forecast cards ------------------------------------------- #
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast Model", mo["model"])
    m2.metric("Walk-forward MAPE", f"{mo['mape']:.2f}%")
    m3.metric("Walk-forward MAE", f"₹{mae/1e5:.2f} L")
    m4.metric("📈 Predicted Revenue (Next Month)", f"₹{next_rev/1e5:.2f} L")

    # ---- Revenue forecast with 95% confidence band ------------------------- #
    fig = _forecast_fig(pm, "forecast_multivariate.csv", "revenue",
                        "Revenue + Occupancy forecast with 95% confidence band", mae)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- Prediction vs Actual (walk-forward) ------------------------------- #
    bt = _load_csv("backtest_multivariate.csv")
    if bt is not None:
        wf = go.Figure()
        wf.add_scatter(x=bt.billing_period, y=bt.actual, name="actual",
                       mode="lines+markers", line=dict(color=C_PRIMARY, width=3))
        wf.add_scatter(x=bt.billing_period, y=bt.multivariate, name="predicted",
                       mode="lines+markers", line=dict(dash="dot", color=C_ACCENT))
        wf.update_layout(title="Prediction vs Actual — Revenue + Occupancy",
                         hovermode="x unified", margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(wf, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- How occupancy relates to revenue ---------------------------------- #
    st.metric("Occupancy–Revenue Correlation",
              f"{mv['occupancy_revenue_corr']:.3f}")
    c1, c2 = st.columns(2)
    sc = _load_csv("occupancy_revenue_scatter.csv")
    if sc is not None:
        f1 = px.scatter(sc, x="occupancy_pct", y="revenue", trendline=None,
                        title=f"Revenue vs Occupancy (r={mv['occupancy_revenue_corr']:.3f})",
                        color_discrete_sequence=[C_PRIMARY])
        f1.add_scatter(x=sc.sort_values("occupancy_pct")["occupancy_pct"],
                       y=sc.sort_values("occupancy_pct")["reg_line"],
                       mode="lines", name="regression",
                       line=dict(color=C_RISK, width=2))
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
    pm = feats["property_month"]
    if "occupancy_pct" in pm.columns:
        pmt = pm.dropna(subset=["occupancy_pct"]).copy()
        pmt["month"] = pmt["billing_period"].astype(str)
        f2 = go.Figure()
        f2.add_scatter(x=pmt.month, y=pmt.revenue, name="revenue",
                       line=dict(color=C_PRIMARY), yaxis="y1")
        f2.add_scatter(x=pmt.month, y=pmt.occupancy_pct, name="occupancy %",
                       line=dict(color=C_WARN), yaxis="y2")
        f2.update_layout(title="Revenue vs Occupancy — monthly trend",
                         yaxis=dict(title="revenue"),
                         yaxis2=dict(title="occupancy %", overlaying="y",
                                     side="right"),
                         hovermode="x unified", margin=dict(l=10, r=10, t=48, b=10))
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)

    # What drives the forecast (feature importance).
    c1, c2 = st.columns(2)
    pi = _load_csv("perm_importance_multivariate.csv")
    if pi is not None:
        pi.columns = ["feature", "importance"]
        f = px.bar(pi.head(12).sort_values("importance"), x="importance",
                   y="feature", orientation="h",
                   title="What drives the forecast",
                   color_discrete_sequence=[C_ACCENT])
        f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
    sh = _load_csv("shap_multivariate.csv")
    if sh is not None:
        sh.columns = ["feature", "mean_abs_shap"]
        f = px.bar(sh.head(12).sort_values("mean_abs_shap"), x="mean_abs_shap",
                   y="feature", orientation="h",
                   title="Key revenue drivers",
                   color_discrete_sequence=[C_PRIMARY])
        f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c2.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)

    # Occupancy Scenario Analysis (±5% occupancy) — centre is the official
    # next-month prediction; the ±5% cards are the existing scenario outputs.
    st.markdown("**Occupancy Scenario Analysis — next-month revenue if occupancy "
                "shifts ±5%**")
    sc_data = mv["scenario"]
    up = sc_data["plus_5pct"] - sc_data["base"]
    dn = sc_data["minus_5pct"] - sc_data["base"]
    s1, s2, s3 = st.columns(3)
    s1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Occupancy −5%</div>"
                f"<div class='kpi-value' style='color:#C0392B'>"
                f"₹{sc_data['minus_5pct']/1e5:.2f} L</div>"
                f"<div class='kpi-label'>{dn/1e5:+.2f} L</div></div>",
                unsafe_allow_html=True)
    s2.markdown(f"<div class='kpi-card'><div class='kpi-label'>📈 Predicted Revenue "
                f"(Next Month)</div><div class='kpi-value'>₹{next_rev/1e5:.2f} L"
                f"</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Occupancy +5%</div>"
                f"<div class='kpi-value' style='color:#27AE60'>"
                f"₹{sc_data['plus_5pct']/1e5:.2f} L</div>"
                f"<div class='kpi-label'>{up/1e5:+.2f} L</div></div>",
                unsafe_allow_html=True)
    forecast = _load_csv("forecast_multivariate.csv")

    if forecast is not None:
         st.download_button(
        "⬇️ Download 6-Month Revenue Forecast",
        forecast.to_csv(index=False),
        "forecast_next_6_months.csv",
        "text/csv"
         )
    


def _multivariate_verdict(mv: dict) -> str:
    ro, mo = mv["revenue_only_best"], mv["multivariate_best"]
    r = mv["occupancy_revenue_corr"]
    tie = abs(ro["mape"] - mo["mape"]) < 0.5
    if mo["mape"] < ro["mape"]:
        head = (f"**The multivariate model wins** (MAPE {mo['mape']:.2f}% vs "
                f"revenue-only {ro['mape']:.2f}%).")
    elif tie:
        head = (f"**MAPE is effectively tied** ({mo['mape']:.2f}% vs "
                f"{ro['mape']:.2f}% — within noise on {mv['n_test_months']} test "
                f"months), but the multivariate model explains more variance "
                f"(R² {mo['r2']:.2f} vs {ro['r2']:.2f}).")
    else:
        head = (f"**Revenue-only edges MAPE** ({ro['mape']:.2f}% vs "
                f"{mo['mape']:.2f}%), though the multivariate model has higher "
                f"R² ({mo['r2']:.2f} vs {ro['r2']:.2f}).")
    why = (f"Occupancy is a strong real driver — occupancy% correlates with revenue "
           f"at **r={r:.3f}**, and lagged occupancy ranks among the top features "
           f"(permutation + SHAP). With only ~{mv['n_train_months_total']} monthly "
           f"rows the one-step MAPE gain is small, but the multivariate model adds "
           f"what revenue-only cannot: **occupancy scenario analysis** and higher "
           f"explained variance. Recommended for planning; time-series stays the "
           f"primary multi-month forecaster.")
    return f"{head}\n\n{why}"


def _revenue_verdict(rmeta: dict, comp) -> str:
    """Explain which revenue model wins and why — from the real metrics."""
    ts, ml = rmeta["ts_mape"], rmeta["ml_mape"]
    winner = rmeta["winner"]
    if winner == "TimeSeries":
        head = (f"**Verdict: the Time-Series model wins** "
                f"(MAPE {ts:.1f}% vs ML {ml:.1f}%).")
        why = ("Its explicit trend + 12-month seasonal structure captures the "
               "yearly PG intake cycle better than an ML model learning from only "
               f"{rmeta['n_train_months_total']} monthly rows.")
    else:
        head = (f"**Verdict: the ML model ({rmeta['ml_model']}) wins** "
                f"(MAPE {ml:.1f}% vs Time-Series {ts:.1f}%).")
        why = ("For one-month-ahead revenue the lagged features (especially last "
               "month's revenue) track the strong month-to-month persistence and "
               "recent growth better than a purely-seasonal naive forecast, which "
               "compares against a year-ago value and misses the 2024→2026 trend. "
               "A linear model beats XGBoost here because ~28 monthly rows are far "
               "too few for gradient boosting — it overfits, while the linear AR "
               "model matches the true low complexity of the signal.")
    caveat = ("With only a few dozen monthly observations both errors carry wide "
              "uncertainty; the Time-Series model remains the primary forecaster "
              "for multi-month horizons, and the ML model is a complementary "
              "one-step cross-check.")
    return f"{head}\n\n{why}\n\n_{caveat}_"


def _segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Business names from real profile: highest paid-LTV segment = Anchor."""
    if profile is None or "ltv_paid" not in profile:
        return {}
    ranked = profile.sort_values("ltv_paid", ascending=False)["segment"].tolist()
    names = {}
    label_bank = ["Anchor Tenants", "Regular Tenants", "Short-stay Tenants",
                  "At-risk Tenants"]
    for i, seg in enumerate(ranked):
        names[int(seg)] = label_bank[i] if i < len(label_bank) else f"Segment {seg}"
    return names


def _exec_summary_items(cleaned, feats) -> list[tuple[str, str, str]]:
    """(icon, label, value) computed from real data + model outputs."""
    pm = feats["property_month"].sort_values("billing_period")
    # Single source of truth: live Ridge multivariate prediction.
    mv = rmv.predict_live(pm) if len(pm) else None
    cur_rev = float(pm["revenue"].iloc[-1])          # from property_month, not invoices
    items = [("💰", "Current Revenue (mo)", f"₹{cur_rev/1e5:.1f} L")]
    if mv is not None:
        items.append((
            "📈",
            "Predicted Revenue (next mo)",
            f"₹{mv['next_month_revenue']/1e5:.2f} L"
        ))
    # Current + predicted occupancy from the same booking-based occupancy series.
    if "occupancy_pct" in pm.columns and pm["occupancy_pct"].notna().any():
        items.append(("🛏️", "Current Occupancy",
                      f"{_current_occupancy_pct(feats):.1f}%"))
    occ_fc = _load_csv("forecast_occupancy_pct.csv")
    if occ_fc is not None and len(occ_fc) and "occupancy_pct" in occ_fc.columns:
        items.append(("🔮", "Predicted Occupancy (next mo)",
                      f"{occ_fc['occupancy_pct'].iloc[0]:.1f}%"))
    if mv is not None:
        items.append(("🔗", "Occupancy↔Revenue Corr",
                      f"{mv['occupancy_revenue_corr']:.2f}"))
    return items


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _kpis(cleaned, feats):
    # Executive Summary KPIs — read ONLY from the latest processed property_month
    # (Phase-2 feature engineering). No old invoice-sum calculations; every value
    # is the current (latest) month of property_month and auto-updates with data.
    pm = feats["property_month"].sort_values("billing_period")
    m = pm.iloc[-1]
    # Financial-year (India Apr->Mar) BILLED revenue — real invoice totals only
    # (ops.financial_year_revenue), auto-detected latest FY. Independent of the
    # property_month values above; existing KPI logic is unchanged.
    fy = ops.financial_year_revenue(cleaned["invoices"])
    # Total historical BILLED revenue — sum of every invoice total_amount across all
    # available months. Real invoice data only; no collection/payment inference.
    total_revenue = float(cleaned["invoices"]["total_amount"].sum())
    return {
        "🏦 Total Revenue": f"₹{total_revenue/1e5:.2f} L",
        f"📅 FY {fy['fy_label']} Revenue": f"₹{fy['revenue']/1e5:.2f} L",
        "🧾 Monthly Expenses": f"₹{m['monthly_expenses']/1e5:.2f} L",
        "📈 Net Revenue": f"₹{m['net_revenue']/1e5:.2f} L",
    }


def _figures(cleaned, feats):
    pm = feats["property_month"].copy()
    pm["month"] = pm["billing_period"].astype(str)
    figs = {}
    figs["revenue"] = px.area(pm, x="month", y="revenue", title="Monthly Revenue",
                              color_discrete_sequence=[C_PRIMARY])
    mt = ops.monthly_revenue_trend(cleaned["invoices"])
    figs["monthly_trend"] = px.line(
        mt, x="month", y="revenue", markers=True,
        title="Monthly Revenue Trend (all months)",
        color_discrete_sequence=[C_ACCENT])
    # Electricity Cost — kept on the PREVIOUS (pre-migration) electricity data,
    # NOT migrated to the new sparse readings. Reads the old monthly elec_cost
    # series (feat_property_month.csv); falls back to live pm only if absent.
    _old_pm = _load_csv("feat_property_month.csv")
    if _old_pm is not None and "elec_cost" in _old_pm.columns:
        _old_pm = _old_pm.copy()
        _old_pm["month"] = _old_pm["billing_period"].astype(str)
        _elec_src = _old_pm[["month", "elec_cost"]]
    else:
        _elec_src = pm[["month", "elec_cost"]]
    figs["electricity"] = px.line(_elec_src, x="month", y="elec_cost", markers=True,
                                  title="Electricity Cost",
                                  color_discrete_sequence=[C_WARN])
    # Bed Lifecycle Mix — sourced from live_bed_snapshot.live_status, the SAME
    # operational source as the Available Beds page, so both pies match exactly.
    _snap = _live_bed_snapshot(cleaned["bookings"])
    bl = _snap["live_status"].value_counts().reset_index()
    bl.columns = ["status", "count"]
    figs["beds"] = px.pie(bl, names="status", values="count",
                          title="Bed Lifecycle Mix", hole=0.45,
                          color="status",
                          color_discrete_map={"Occupied": C_PRIMARY,
                                              "Notice": C_RISK,
                                              "Notice-Booked": C_MED,
                                              "Booked": C_ACCENT,
                                              "Vacant": C_WARN,
                                              "Inactive": "#7f8c8d"})
    tk = cleaned["tickets"]["issue_type"].value_counts().head(10).reset_index()
    tk.columns = ["issue_type", "count"]
    figs["tickets"] = px.bar(tk, x="count", y="issue_type", orientation="h",
                             title="Top Maintenance Issues",
                             color_discrete_sequence=[C_ACCENT])
    for f in figs.values():
        f.update_layout(margin=dict(l=10, r=10, t=48, b=10),
                        title_font_size=15)
    return figs


def _forecast_fig(pm, csv_name, col, title, mae=None):
    fc = _load_csv(csv_name)
    if fc is None:
        return None
    fig = go.Figure()
    fig.add_scatter(x=pm["billing_period"].astype(str), y=pm[col], name="actual",
                    mode="lines+markers", line=dict(color=C_PRIMARY))
    fig.add_scatter(x=fc["billing_period"], y=fc[col], name="forecast",
                    mode="lines+markers", line=dict(dash="dash", color=C_RISK))
    if mae is not None and mae == mae:
        band = 1.96 * mae   # ~95% interval from real walk-forward errors
        fig.add_scatter(x=fc["billing_period"], y=fc[col] + band,
                        mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc["billing_period"], y=(fc[col] - band).clip(lower=0),
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(231,111,81,0.15)",
                        name="95% confidence (walk-forward)")
    fig.update_layout(title=title, hovermode="x unified",
                      margin=dict(l=10, r=10, t=48, b=10))
    return fig


def _backtest_fig(csv_name, title):
    bt = _load_csv(csv_name)
    if bt is None:
        return None
    fig = go.Figure()
    fig.add_scatter(x=bt["billing_period"], y=bt["actual"], name="actual",
                    mode="lines+markers", line=dict(color=C_PRIMARY))
    fig.add_scatter(x=bt["billing_period"], y=bt["predicted"], name="predicted",
                    mode="lines+markers", line=dict(dash="dot", color=C_ACCENT))
    fig.update_layout(title=title, hovermode="x unified",
                      margin=dict(l=10, r=10, t=48, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title=APP_TITLE, page_icon="🏢", layout="wide")
    st.markdown("""
    <style>
      .kpi-card {border:1px solid rgba(128,128,128,.25); border-radius:12px;
                 padding:14px 16px; text-align:center; height:100%;}
      .kpi-label {font-size:.78rem; opacity:.75; margin-bottom:4px;
                  white-space:nowrap;}
      .kpi-value {font-size:1.35rem; font-weight:700;}
      .badge {display:inline-block; padding:2px 10px; border-radius:10px;
              color:white; font-size:.75rem; font-weight:600;}
      .badge-High {background:#C0392B;} .badge-Medium {background:#E67E22;}
      .badge-Low {background:#27AE60;}
      .summary-strip {border-left:4px solid #2A9D8F; padding:8px 14px;
                      margin:4px 0; border-radius:4px;
                      background:rgba(42,157,143,.07);}
      div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);
                                   border-radius:12px; padding:10px;}
    </style>""", unsafe_allow_html=True)

    st.title(f"🏢 {APP_TITLE}")

    cleaned, feats = _data()
    pm = feats["property_month"].copy()
    figs = _figures(cleaned, feats)
    fs = _load_csv("forecast_summary.csv")

    # ---- Auto Executive Summary strip (real model outputs only) ------------ #
    with st.container():
        items = _exec_summary_items(cleaned, feats)
        cols = st.columns(len(items))
        for c, (icon, label, value) in zip(cols, items):
            c.markdown(f"<div class='kpi-card'><div class='kpi-label'>{icon} "
                       f"{label}</div><div class='kpi-value'>{value}</div></div>",
                       unsafe_allow_html=True)
    st.markdown("")

    # Sidebar navigation — render ONLY the selected page (single-page behavior).
    # (Previously st.tabs, which emits every page's content on each run.)
    PAGES = [
        "📊 Executive Summary", "📈 Revenue Forecast", "🛏️ Occupancy Forecast",
        "🏠 Apartment-wise Forecast", "💰 Financial Overview",
        "👥 Tenant Segmentation", "💡 AI Recommendations",
        "📦 Asset Management", "🚪 Available Beds", "🔧 Maintenance",
        "🏆 Apartment Performance", "📤 Notice & Exit", "📊 Business Insights"]
    page = st.sidebar.radio("Navigate", PAGES, index=0)

    # 1) Executive Summary ---------------------------------------------------- #
    if page == PAGES[0]:
        kpis = _kpis(cleaned, feats)
        cols = st.columns(len(kpis))
        for col, (k, v) in zip(cols, kpis.items()):
            col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{k}</div>"
                         f"<div class='kpi-value'>{v}</div></div>",
                         unsafe_allow_html=True)
        st.markdown("")

        # ---- Financial Year selector (India Apr->Mar) --------------------- #
        # Revenue per FY from real invoice totals only (ops.revenue_by_financial_year).
        # The dropdown lists every FY present in the invoice data, so it stays
        # correct automatically as the database grows. Existing KPI/revenue
        # calculations above are not touched.
        fy_tbl = ops.revenue_by_financial_year(cleaned["invoices"])
        if len(fy_tbl):
            st.markdown("#### Financial Year Revenue")
            fy_labels = fy_tbl["fy"].tolist()          # ascending
            sel = st.selectbox("Select financial year (April–March)",
                               fy_labels[::-1], index=0, key="exec_fy_select")
            pos = fy_labels.index(sel)
            row = fy_tbl.iloc[pos]
            prev = fy_tbl.iloc[pos - 1] if pos > 0 else None
            sel_rev = float(row["revenue"])
            prev_rev = float(prev["revenue"]) if prev is not None else None
            in_prog = bool(row["in_progress"])

            f1, f2, f3 = st.columns(3)
            f1.metric(f"FY {sel} Revenue" + (" (in progress)" if in_prog else ""),
                      f"₹{sel_rev/1e5:.2f} L")
            if prev_rev:
                yoy = (sel_rev - prev_rev) / prev_rev * 100
                f2.metric(f"FY {prev['fy']} Revenue", f"₹{prev_rev/1e5:.2f} L")
                f3.metric("YoY Growth", f"{yoy:+.1f}%", delta=f"{yoy:+.1f}%")
            else:
                f2.metric("Previous FY Revenue", "—")
                f3.metric("YoY Growth", "—")

            note = (f"FY {sel}: {int(row['months'])} month(s), "
                    f"{int(row['invoices'])} invoices. Revenue = sum of invoice "
                    "total_amount (April→March).")
            if in_prog:
                note += (" This FY is still in progress, so YoY vs a full prior "
                         "year is not directly comparable.")
            st.caption(note)

        c1, c2 = st.columns(2)
        c1.plotly_chart(figs["monthly_trend"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["electricity"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c1.plotly_chart(figs["beds"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["tickets"], use_container_width=True,
                        config=PLOTLY_CONFIG)

        # ---- FY Revenue Comparison (additional visualization only) --------- #
        # Reuses the same invoice FY aggregation already computed above for the
        # selector. No KPI/dropdown/calculation changes — chart only. New FYs
        # appear automatically as invoice data grows.
        if len(fy_tbl):
            st.markdown("#### Financial Year Revenue Comparison")
            fy_chart = fy_tbl.copy()
            fy_chart["revenue_lakh"] = (fy_chart["revenue"] / 1e5).round(2)
            fy_chart["fy_axis"] = "FY " + fy_chart["fy"].astype(str)
            fy_chart.loc[fy_chart["in_progress"], "fy_axis"] = (
                fy_chart.loc[fy_chart["in_progress"], "fy_axis"] + " (In Progress)")
            fy_chart["Status"] = np.where(
                fy_chart["in_progress"], "In Progress", "Complete")
            fig_fy = px.bar(
                fy_chart, x="fy_axis", y="revenue_lakh", color="Status",
                title="Financial Year Revenue Comparison",
                labels={"fy_axis": "Financial Year",
                        "revenue_lakh": "Total Revenue (₹ Lakhs)"},
                text="revenue_lakh",
                color_discrete_map={"Complete": C_PRIMARY,
                                    "In Progress": C_WARN})
            fig_fy.update_traces(textposition="outside")
            fig_fy.update_layout(margin=dict(l=10, r=10, t=48, b=10),
                                 title_font_size=15,
                                 xaxis_title="Financial Year",
                                 yaxis_title="Total Revenue (₹ Lakhs)",
                                 legend_title_text="")
            st.plotly_chart(fig_fy, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("Actual billed revenue by India financial year (April→March), "
                       "sum of invoice total_amount. Incomplete latest FY is marked "
                       "In Progress. Updates automatically when new invoice years "
                       "appear in the data.")

    # 2) Revenue Forecast ------------------------------------------------------ #
    if page == PAGES[1]:
       
        # ---- Third model: multivariate revenue + occupancy ----------------- #
        st.markdown("## Revenue Forecast")
        st.markdown("### Revenue Forecast based on Historical Occupancy")
        st.caption("Revenue prediction is based on historical revenue and historical occupancy "
        "features from the real booking/stay dataset. Only lagged occupancy values "
        "are used to avoid data leakage.")
        _render_multivariate(st, feats)

    # 3) Occupancy Forecast ---------------------------------------------------- #
    if page == PAGES[2]:
        st.subheader("Occupancy Forecast (Booking-based)")
        st.caption("Monthly occupancy from the processed property_month. "
                   f"Occupancy % = Occupied Beds / Total Beds ({config.TOTAL_BEDS}).")

        # ---- Monthly occupancy from property_month (single source) --------- #
        opm = feats["property_month"].sort_values("billing_period").copy()
        opm["month"] = opm["billing_period"].astype(str)
        olast = opm.iloc[-1]
        o1, o2, o3 = st.columns(3)
        o1.metric(f"Occupancy % ({olast['month']})",
                  f"{olast['occupancy_pct']:.1f}%")
        # Presentation-only: show the count as "N Beds" (still read dynamically from
        # olast["occupied_beds"]); explanatory text shows inside the same card,
        # directly below the value (via the metric delta slot, no arrow/colour).
        # The underlying value is unchanged.
        o2.metric("Occupied Beds", f"{int(olast['occupied_beds'])} Beds",
                  delta="Based on latest monthly booking occupancy",
                  delta_color="off")
        o3.metric("Vacant Beds",
                  f"{int(config.TOTAL_BEDS - olast['occupied_beds'])}")

        def _occ_trend(col, title, color):
            f = px.line(opm, x="month", y=col, markers=True, title=title,
                        color_discrete_sequence=[color])
            f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            return f

        tr1, tr2 = st.columns(2)
        tr1.plotly_chart(_occ_trend("occupancy_pct", "Occupancy % Trend", C_PRIMARY),
                         use_container_width=True, config=PLOTLY_CONFIG)
        tr2.plotly_chart(_occ_trend("occupied_beds", "Occupied Beds Trend", C_ACCENT),
                         use_container_width=True, config=PLOTLY_CONFIG)
        st.plotly_chart(_occ_trend("active_tenants", "Active Tenants Trend", C_WARN),
                        use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### Occupancy Forecast")
        occ = _load_csv("forecast_occupancy_pct.csv")
        mae_t = None
        if fs is not None:
            row = fs[fs.series == "occupied_beds"]
            if len(row):
                mae_t = float(row.MAE.iloc[0])
                c1, c2, c3 = st.columns(3)
                c1.metric("Forecast method", row.method.iloc[0])
                c2.metric("Walk-forward MAPE", f"{row.MAPE.iloc[0]:.1f}%")
                if occ is not None and len(occ):
                    # Presentation-only: show the predicted occupancy % only (no bed
                    # count). Value stays dynamic from the forecast pipeline.
                    c3.metric("Next month Occupancy",
                              f"{occ.iloc[0]['occupancy_pct']:.1f}%")
        fig = _forecast_fig(pm, "forecast_occupancy_pct.csv", "occupied_beds",
                            "Occupied Beds Forecast", mae_t)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        bfig = _backtest_fig("backtest_occupied_beds.csv",
                             "Prediction vs Actual (Occupied Beds)")
        if bfig:
            st.plotly_chart(bfig, use_container_width=True, config=PLOTLY_CONFIG)
        fc = _load_csv("forecast_occupancy_pct.csv")
        if fc is not None:
            st.dataframe(fc, use_container_width=True)
            st.download_button("⬇️ Export occupancy forecast CSV",
                               fc.to_csv(index=False),
                               "forecast_occupancy_pct.csv", "text/csv")

    # 4) Apartment-wise Forecast ----------------------------------------------- #
    if page == PAGES[3]:
        st.subheader("Apartment-wise electricity forecast")
        st.caption("Electricity is the only real apartment × month series in the "
                   "data (invoices carry no apartment code), so apartment-level "
                   "forecasting covers electricity units and amount.")
        apt = _load_csv("forecast_apartment_summary.csv")
        if apt is not None:
            sel = st.multiselect("Filter apartments",
                                 sorted(apt["apartment_code"].unique()))
            view = apt[apt.apartment_code.isin(sel)] if sel else apt
            fig = px.bar(view.head(20), x="apartment_code", y="next_month_amount",
                         color="amount_mape", color_continuous_scale="RdYlGn_r",
                         title="Next-month electricity amount "
                               "(colour = backtest MAPE %)",
                         labels={"next_month_amount": "₹ next month"})
            fig.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.dataframe(view, use_container_width=True, height=320)
            full = _load_csv("forecast_apartment_electricity.csv")
            if full is not None:
                st.download_button("⬇️ Export apartment forecast CSV",
                                   full.to_csv(index=False),
                                   "forecast_apartment_electricity.csv", "text/csv")
        else:
            st.info("Run: python -m src.apartment_forecasting")

    # 5) Financial Overview ---------------------------------------------------- #
    if page == PAGES[4]:
        st.subheader("Financial Overview")
        st.caption("Monthly financial metrics from the processed property_month "
                   "(Phase-2 feature engineering) — the single source. Charts update "
                   "automatically as new production months are added.")
        fpm = feats["property_month"].sort_values("billing_period").copy()
        fpm["month"] = fpm["billing_period"].astype(str)
        flast = fpm.iloc[-1]

        # Latest-month KPIs (from property_month — matches Executive Summary).
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"Revenue ({flast['month']})", f"₹{flast['revenue']/1e5:.2f} L")
        c2.metric("Collections", f"₹{flast['collections']/1e5:.2f} L")
        c3.metric("Outstanding", f"₹{flast['outstanding_balance']/1e5:.2f} L")
        c4.metric("Expenses", f"₹{flast['monthly_expenses']/1e5:.2f} L")
        c5.metric("Net Revenue", f"₹{flast['net_revenue']/1e5:.2f} L")

        def _fin_trend(col, title, color):
            f = px.line(fpm, x="month", y=col, markers=True, title=title,
                        color_discrete_sequence=[color])
            f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            return f

        t1, t2 = st.columns(2)
        t1.plotly_chart(_fin_trend("revenue", "Monthly Revenue Trend", C_PRIMARY),
                        use_container_width=True, config=PLOTLY_CONFIG)
        t2.plotly_chart(_fin_trend("collections", "Collections Trend", C_ACCENT),
                        use_container_width=True, config=PLOTLY_CONFIG)
        t3, t4 = st.columns(2)
        t3.plotly_chart(_fin_trend("monthly_expenses", "Monthly Expenses Trend",
                                   C_WARN),
                        use_container_width=True, config=PLOTLY_CONFIG)
        t4.plotly_chart(_fin_trend("net_revenue", "Net Revenue Trend", C_PRIMARY),
                        use_container_width=True, config=PLOTLY_CONFIG)

        # ---- Revenue Forecast: single source of truth = live Ridge multivariate #
        st.markdown("#### Revenue Forecast")
        mvfin = rmv.predict_live(feats["property_month"]) \
            if len(feats["property_month"]) else None
        if mvfin is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Forecast model", f"{mvfin['model']} multivariate")
            m2.metric("Walk-forward MAPE", f"{mvfin['mape']:.2f}%")
            m3.metric("MAE / RMSE",
                      f"₹{mvfin['mae']:,.0f} / ₹{mvfin['rmse']:,.0f}")
            st.caption(f"Next-month revenue forecast: "
                       f"₹{mvfin['next_month_revenue']/1e5:.2f} L "
                       f"(model: {mvfin['model']} multivariate, walk-forward MAPE "
                       f"{mvfin['mape']:.1f}%). Single source of truth — identical "
                       "to the Revenue Forecast page and AI Recommendations.")
        # Time-series cross-check (Holt-Winters) kept available for reference.
        comp = _load_csv("revenue_forecast_comparison.csv")
        if comp is not None:
            st.markdown("**Time-series cross-check — Holt-Winters vs ML "
                        "(from forecasting.py)**")
            st.dataframe(comp, use_container_width=True)

        st.download_button("⬇️ Export monthly financials CSV",
                           fpm[["month", "revenue", "collections",
                                "outstanding_balance", "monthly_expenses",
                                "net_revenue"]].to_csv(index=False),
                           "financial_overview_monthly.csv", "text/csv")

    # 6) Tenant Segmentation --------------------------------------------------- #
    if page == PAGES[5]:
        st.subheader("Tenant segments — real billing behaviour")
        # Headline metrics straight from tenant_features (Phase-2 output).
        tf = feats["tenant_features"]
        tk1, tk2, tk3 = st.columns(3)
        tk1.metric("👥 Total Tenants", f"{len(tf):,}")
        tk2.metric("💰 Avg Lifetime Value", f"₹{tf['ltv_paid'].mean()/1e5:.2f} L")
        tk3.metric("🏠 Avg Rent / Tenant", f"₹{tf['avg_rent'].mean():,.0f}")
        st.caption("Tenant count, average revenue and lifetime value from "
                   "tenant_features; segment groupings from segmentation.py "
                   "(segmentation logic unchanged).")
        seg = _load_csv("tenant_segments_profile.csv")
        if seg is not None:
            names = _segment_names(seg)
            seg["segment_name"] = seg["segment"].map(names)
            c1, c2 = st.columns(2)
            f1 = px.bar(seg, x="segment_name", y="ltv_paid",
                        title="Average lifetime value (paid) per tenant",
                        color="segment_name",
                        color_discrete_sequence=[C_PRIMARY, C_ACCENT])
            f2 = px.bar(seg, x="segment_name", y="n_tenants",
                        title="Tenants per segment", color="segment_name",
                        color_discrete_sequence=[C_PRIMARY, C_ACCENT])
            for f in (f1, f2):
                f.update_layout(showlegend=False,
                                margin=dict(l=10, r=10, t=48, b=10))
            c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
            c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
            st.dataframe(seg.set_index("segment_name"), use_container_width=True)

            st.markdown("**Business interpretation** (computed from the profile):")
            for _, r in seg.iterrows():
                st.markdown(
                    f"- **{r['segment_name']}** — {int(r['n_tenants'])} tenants, "
                    f"avg tenure {r['tenure_months']:.0f} months, avg rent "
                    f"₹{r['avg_rent']:,.0f}, lifetime value ₹{r['ltv_paid']/1e5:.2f} "
                    f"Lakhs, unpaid ratio {r['unpaid_ratio']:.0%}. "
                    + ("Core revenue base — protect with priority service and "
                       "renewal incentives." if r["ltv_paid"] == seg.ltv_paid.max()
                       else "Growth pool — convert to long-stay via upgrade offers "
                            "and consistent service quality."))
            segs_full = _load_csv("tenant_segments.csv")
            if segs_full is not None:
                segs_full["segment_name"] = segs_full["segment"].map(names)
                st.download_button("⬇️ Export segment assignments CSV",
                                   segs_full.to_csv(index=False),
                                   "tenant_segments.csv", "text/csv")

    # 7) AI Recommendations --------------------------------------------------- #
    if page == PAGES[6]:
        st.subheader("💡 AI Recommendations")
        st.caption("Your live business assistant. It summarises the latest structured "
                   "output of every dashboard page (Financial Overview, Occupancy, "
                   "Notice & Exit, Maintenance, Available Beds and more) — it does not "
                   "compute its own numbers, so it updates automatically whenever any "
                   "page's data changes.")
        # Single source of truth: the recommendation engine reads each page's
        # analytics output and only summarises it (no duplicated business logic).
        outputs, cards = rec.recommend(cleaned, feats)
        n_high = sum(c["priority"] == "High" for c in cards)
        n_med = sum(c["priority"] == "Medium" for c in cards)
        n_low = sum(c["priority"] == "Low" for c in cards)

        st.markdown("#### Business AI Summary")
        s1, s2, s3 = st.columns(3)
        s1.metric("🔴 High Priority", n_high)
        s2.metric("🟠 Medium Priority", n_med)
        s3.metric("🟢 Low Priority", n_low)
        st.markdown("")

        # Small Electricity Alert card — the one kept operational alert.
        ea = outputs.get("electricity_alert")
        if ea is not None and len(ea):
            for _, r in ea.iterrows():
                st.markdown(
                    f"<div class='summary-strip'>⚡ <b>Electricity Alert</b><br>"
                    f"Apartment : <b>{r['apartment_code']}</b><br>"
                    f"Status : Vacant Apartment Using Power<br>"
                    f"Units Consumed : {int(round(float(r['units_consumed'])))}<br>"
                    f"Expected Usage : 0</div>", unsafe_allow_html=True)
            st.markdown("")

        if not cards:
            st.info("No recommendations — no dashboard page reported an actionable "
                    "metric.")
        for c in cards:
            st.markdown(
                f"<div class='summary-strip'>"
                f"<span class='badge badge-{c['priority']}'>{c['priority']}</span>"
                f"&nbsp;&nbsp;<b>{c['icon']} {c['category']}</b>"
                f"<span style='opacity:.55'> · from {c['source']}</span><br>"
                f"{c['recommendation']}<br>"
                f"<span style='opacity:.8'>💼 <b>Expected business impact:</b> "
                f"{c['impact']}</span></div>", unsafe_allow_html=True)

        if cards:
            dfc = pd.DataFrame(cards)[["priority", "category", "source",
                                       "recommendation", "impact"]]
            st.download_button("⬇️ Export AI recommendations CSV",
                               dfc.to_csv(index=False), "ai_recommendations.csv",
                               "text/csv")

    # 8) Asset Management ------------------------------------------------------ #
    if page == PAGES[7]:
        st.subheader("Asset Management")
        a = ops.assets_summary(cleaned["assets"])
        _kpi_cards(st, [
            ("📦 Total Assets", f"{a['total']:,}"),
            ("🗂️ Categories", f"{len(a['by_category'])}"),
            ("🔧 Asset Types", f"{len(a['by_type'])}"),
            ("🟢 Allocated",
             f"{int(a['by_status'].loc[a['by_status']['status']=='allocated','count'].sum())}"),
        ])
        st.info("ℹ️ Warranty alerts, assets-by-apartment and Active/Damaged splits "
                "are **not shown** — `warranty_expiry` is 0.1% populated, "
                "`apartment_code` is 100% null, and `condition` has no 'damaged' "
                "value (only good/new). See dataset audit.")
        c1, c2 = st.columns(2)
        f1 = px.pie(a["by_category"], names="category", values="count", hole=0.45,
                    title="Assets by Category",
                    color_discrete_sequence=px.colors.sequential.Teal)
        f2 = px.bar(a["by_type"].sort_values("count"), x="count", y="type",
                    orientation="h", title="Assets by Type (top 15)",
                    color_discrete_sequence=[C_ACCENT])
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        f3 = px.bar(a["by_status"], x="status", y="count", title="Assets by Status",
                    color="status", color_discrete_sequence=[C_PRIMARY, C_WARN])
        f4 = px.bar(a["by_condition"], x="condition", y="count",
                    title="Assets by Condition",
                    color="condition", color_discrete_sequence=[C_PRIMARY, C_ACCENT])
        for f in (f3, f4):
            f.update_layout(showlegend=False, margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f4, use_container_width=True, config=PLOTLY_CONFIG)
        if a["purchase_timeline"] is not None:
            st.markdown(f"**Assets purchased by month** "
                        f"(only {a['purchase_coverage']:,} of {a['total']:,} assets "
                        f"have a purchase_date — {a['purchase_coverage']/a['total']*100:.0f}%)")
            f5 = px.bar(a["purchase_timeline"], x="month", y="assets",
                        title="Assets Purchased by Month (subset with dates)",
                        color_discrete_sequence=[C_WARN])
            f5.update_layout(margin=dict(l=10, r=10, t=48, b=10))
            st.plotly_chart(f5, use_container_width=True, config=PLOTLY_CONFIG)
        cats = st.multiselect("Filter category", sorted(cleaned["assets"]
                                                        ["category"].unique()))
        tbl = a["table"]
        if cats:
            tbl = tbl[tbl["category"].isin(cats)]
        st.dataframe(tbl, use_container_width=True, height=300)
        st.download_button("⬇️ Export assets CSV", tbl.to_csv(index=False),
                           "assets.csv", "text/csv")
        st.caption(f"**Summary:** {a['total']:,} assets, led by "
                   f"{a['by_category'].iloc[0]['category']} "
                   f"({a['by_category'].iloc[0]['count']:,}). "
                   f"{int(a['by_status'].loc[a['by_status'].status=='inventory','count'].sum())} "
                   f"in inventory (not yet allocated).")

    # 9) Available Beds -------------------------------------------------------- #
    if page == PAGES[8]:
        st.subheader("Available Beds")
        # Single source for cards + charts: six mutually exclusive live states.
        live_bed_snapshot = _live_bed_snapshot(cleaned["bookings"])
        live = _live_bed_kpis_from_snapshot(live_bed_snapshot)
        total_beds = live["total_beds"]
        operational_beds = live["operational_beds"]
        occupied_beds = live["occupied_beds"]
        notice_beds = live["notice_beds"]
        notice_booked_beds = live["notice_booked_beds"]
        booked_beds = live["booked_beds"]
        inactive_beds = live["inactive_beds"]
        vacant = live["vacant"]
        occ_pct = live["occ_pct"]
        vac_opp = float(
            live_bed_snapshot.loc[live_bed_snapshot["is_vacant"] == 1,
                                  "current_rate"].fillna(0).sum())
        _kpi_cards(st, [
            ("🛏️ Total Beds", f"{total_beds}"),
            ("⚙️ Operational", f"{operational_beds}"),
            ("👤 Occupied", f"{occupied_beds}"),
            ("📢 Notice", f"{notice_beds}"),
            ("🔁 Notice-Booked", f"{notice_booked_beds}"),
            ("📅 Booked", f"{booked_beds}"),
            ("🚫 Inactive", f"{inactive_beds}"),
            ("🚪 Vacant", f"{vacant}"),
            ("📊 Occupancy %", f"{occ_pct:.1f}%"),
        ])
        st.caption("Live bed snapshot — six mutually exclusive states "
                   "(Occupied / Notice / Notice-Booked / Booked / Inactive / Vacant). "
                   f"Vacant = Operational ({operational_beds}) − Occupied − Notice − "
                   "Notice-Booked − Booked. Cards and charts share this source.")
        c1, c2 = st.columns(2)
        lc = live_bed_snapshot["live_status"].value_counts().reset_index()
        lc.columns = ["status", "count"]
        f1 = px.pie(lc, names="status", values="count", hole=0.45,
                    title="Bed Lifecycle Mix", color="status",
                    color_discrete_map={"Occupied": C_PRIMARY, "Notice": C_RISK,
                                        "Notice-Booked": C_MED, "Booked": C_ACCENT,
                                        "Vacant": C_WARN, "Inactive": "#7f8c8d"})
        op = live_bed_snapshot[live_bed_snapshot["live_status"] != "Inactive"]
        blk = (op.groupby("block", as_index=False)
               .agg(total=("bed_id", "count"),
                    vacant=("is_vacant", "sum")))
        blk["vacancy_pct"] = (blk["vacant"] / blk["total"] * 100).round(1)
        f2 = px.bar(blk.sort_values("vacancy_pct", ascending=False),
                    x="block", y="vacancy_pct", title="Vacancy % by Block",
                    color="vacancy_pct", color_continuous_scale="OrRd")
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        _tenant_states = ["Occupied", "Notice", "Notice-Booked"]
        apt = (live_bed_snapshot.groupby("apartment_code", as_index=False)
               .agg(vacant_beds=("is_vacant", "sum"),
                    total_beds=("bed_id", "count"),
                    inactive_beds=("is_inactive", "sum"),
                    occupied_now=("live_status",
                                  lambda s: int(s.isin(_tenant_states).sum()))))
        apt["operational_beds"] = apt["total_beds"] - apt["inactive_beds"]
        apt["occupancy_pct"] = np.where(
            apt["operational_beds"] > 0,
            (apt["occupied_now"] / apt["operational_beds"] * 100).round(1),
            0.0)
        apt_v = apt[apt["vacant_beds"] > 0].sort_values(
            "vacant_beds", ascending=False)
        f3 = px.bar(apt_v, x="apartment_code", y="vacant_beds",
                    title="Vacant Beds by Apartment",
                    color_discrete_sequence=[C_RISK])
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)

        occ_map = apt.set_index("apartment_code")["occupancy_pct"]
        tbl = live_bed_snapshot[["apartment_code", "bed_code", "live_status",
                                 "current_rate", "is_vacant"]].copy()
        tbl["occupancy_pct"] = tbl["apartment_code"].map(occ_map)
        only_vac = st.checkbox("Show only vacant beds", value=True)
        view = (tbl[tbl["is_vacant"] == 1] if only_vac else tbl)[
            ["apartment_code", "bed_code", "live_status", "current_rate",
             "occupancy_pct"]].rename(columns={
                "apartment_code": "Apartment Code", "bed_code": "Bed Code",
                "live_status": "Bed Status", "current_rate": "Monthly Rent",
                "occupancy_pct": "Occupancy %"})
        st.dataframe(view, use_container_width=True, height=320)
        st.download_button("⬇️ Export beds CSV", view.to_csv(index=False),
                           "available_beds.csv", "text/csv")
        st.caption(f"**Summary:** {vacant} of {operational_beds} operational beds vacant "
                   f"({vacant/operational_beds*100:.1f}%); {inactive_beds} inactive. "
                   f"Vacant revenue opportunity ₹{vac_opp/1e5:.2f} L/mo.")

    # 10) Maintenance Performance ----------------------------------------------- #
    if page == PAGES[9]:
        st.subheader("Maintenance Performance")
        ms = ops.maintenance_summary(cleaned["tickets"])
        _kpi_cards(st, [
            ("🔧 Total Tickets", f"{ms['total']:,}"),
            ("🟠 Open", f"{ms['open']}"),
            ("✅ Closed", f"{ms['closed']:,}"),
            ("⏱️ Avg Resolution", f"{ms['avg_resolution_hours']:.0f} h"),
            ("⛔ SLA Breached",
             f"{ms['sla_breached']} ({ms['sla_breach_pct']}%)"),
        ])
        c1, c2 = st.columns(2)
        f1 = px.pie(ms["by_status"], names="status", values="count", hole=0.45,
                    title="Ticket Status", color_discrete_sequence=
                    px.colors.sequential.Teal)
        f2 = px.bar(ms["by_priority"], x="priority", y="count",
                    title="Ticket Priority", color="priority",
                    color_discrete_map={"high": C_HIGH, "medium": C_MED,
                                        "low": C_LOW, "urgent": "#7b241c"})
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(showlegend=False, margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        f3 = px.bar(ms["by_issue"].sort_values("count"), x="count", y="issue_type",
                    orientation="h", title="Issue Type Analysis",
                    color_discrete_sequence=[C_ACCENT])
        f4 = px.line(ms["monthly"], x="month", y="tickets", markers=True,
                     title="Monthly Ticket Trend",
                     color_discrete_sequence=[C_PRIMARY])
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f4.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f4, use_container_width=True, config=PLOTLY_CONFIG)
        f5 = px.bar(ms["by_apartment"].head(20), x="apartment_code", y="complaints",
                    title="Apartment-wise Complaints (top 20)",
                    color="complaints", color_continuous_scale="OrRd")
        f5.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f5, use_container_width=True, config=PLOTLY_CONFIG)
        st.download_button("⬇️ Export apartment complaints CSV",
                           ms["by_apartment"].to_csv(index=False),
                           "apartment_complaints.csv", "text/csv")
        st.caption(f"**Summary:** {ms['open']} open of {ms['total']:,} tickets; "
                   f"average resolution {ms['avg_resolution_hours']:.0f} hours; "
                   f"SLA breached on {ms['sla_breach_pct']}%. Top issue: "
                   f"{ms['by_issue'].iloc[0]['issue_type']} "
                   f"({ms['by_issue'].iloc[0]['count']}).")

    # 11) Apartment Performance ------------------------------------------------- #
    if page == PAGES[10]:
        st.subheader("Apartment Performance & Health Score")
        ap = ops.apartment_performance(cleaned["electricity"], cleaned["tickets"],
                                       cleaned["beds_snapshot"])
        _kpi_cards(st, [
            ("🏠 Apartments", f"{len(ap)}"),
            ("🏆 Healthiest", f"{ap.iloc[0]['apartment_code']}"),
            ("⚠️ Lowest Health", f"{ap.iloc[-1]['apartment_code']}"),
            ("🔧 Most Complaints",
             f"{ap.sort_values('complaints', ascending=False).iloc[0]['apartment_code']}"),
        ])
        st.info("ℹ️ Apartment-wise **revenue/collection are not shown** — invoices "
                "carry no apartment_code (UUID tenant only). Health score blends "
                "real apartment-keyed metrics: complaints (45%), vacant beds (35%), "
                "electricity cost (20%); higher = healthier.")
        top = st.slider("Show top/bottom N apartments by health", 5, len(ap), 15)
        f1 = px.bar(ap.head(top), x="apartment_code", y="health_score",
                    title=f"Top {top} Apartments by Health Score",
                    color="health_score", color_continuous_scale="RdYlGn")
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        f2 = px.bar(ap.sort_values("elec_cost", ascending=False).head(15),
                    x="apartment_code", y="elec_cost",
                    title="Electricity Cost Ranking (top 15)",
                    color_discrete_sequence=[C_WARN])
        f3 = px.bar(ap.sort_values("complaints", ascending=False).head(15),
                    x="apartment_code", y="complaints",
                    title="Complaint Hotspots (top 15)",
                    color_discrete_sequence=[C_RISK])
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        st.dataframe(ap[["apartment_code", "health_score", "complaints", "vacant",
                         "elec_cost", "avg_units", "complaint_rank", "elec_rank"]]
                     .round(1), use_container_width=True, height=320)
        st.download_button("⬇️ Export apartment performance CSV",
                           ap.to_csv(index=False), "apartment_performance.csv",
                           "text/csv")
        st.caption(f"**Summary:** {ap.iloc[0]['apartment_code']} is healthiest "
                   f"(score {ap.iloc[0]['health_score']}); "
                   f"{ap.sort_values('complaints', ascending=False).iloc[0]['apartment_code']} "
                   f"has the most complaints "
                   f"({int(ap['complaints'].max())}).")

    # 12) Notice & Exit --------------------------------------------------------- #
    if page == PAGES[11]:
        st.subheader("Notice & Exit Analytics")
        na = ops.notice_analytics(cleaned["notices"])   # full history (for charts)
        # KPIs use ONLY active/upcoming notices (exit still in the future).
        nn = cleaned["notices"].copy()
        _ex = pd.to_datetime(nn["estimated_exit_date"], utc=True, errors="coerce")
        _now = pd.Timestamp.now(tz="UTC")
        active_mask = _ex >= _now
        active_notices = int(active_mask.sum())
        upcoming_30 = int(((_ex >= _now) & (_ex <= _now + pd.Timedelta(days=30))).sum())
        rev_at_risk = float(nn.loc[active_mask, "monthly_rental"].sum())
        _kpi_cards(st, [
            ("📋 Active Notices", f"{active_notices}"),
            ("🚪 Upcoming Exits (30d)", f"{upcoming_30}"),
            ("💸 Expected Revenue at Risk", f"₹{rev_at_risk/1e5:.2f} L"),
            ("📅 Avg Notice Period", f"{na['avg_notice_days']:.0f} days"),
        ])
        st.caption(f"KPIs count only active/upcoming notices ({active_notices} of "
                   f"{len(nn)} total). Completed notices are excluded from KPIs but "
                   "remain in the history charts and tables below.")
        st.info("ℹ️ **Notice reasons are not shown** — the notices dataset has no "
                "reason/remarks column.")
        c1, c2 = st.columns(2)
        f1 = px.bar(na["monthly"], x="notice_month", y="notices",
                    title="Monthly Notice Trend",
                    color_discrete_sequence=[C_RISK])
        f2 = px.bar(na["exit_month"], x="exit_month", y="revenue_impact",
                    title="Revenue Impact by Exit Month (₹)",
                    color_discrete_sequence=[C_WARN])
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        f3 = px.bar(na["by_apartment"], x="apartment_code", y="notices",
                    title="Apartment-wise Notice Count",
                    color="notices", color_continuous_scale="OrRd")
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("**Upcoming vacating beds**")
        st.dataframe(na["upcoming_table"], use_container_width=True, height=280)
        st.download_button("⬇️ Export upcoming exits CSV",
                           na["upcoming_table"].to_csv(index=False),
                           "upcoming_exits.csv", "text/csv")
        st.caption(f"**Summary:** {na['upcoming_exits']} tenants scheduled to vacate; "
                   f"₹{na['monthly_revenue_impact']/1e5:.2f} L monthly rent at stake "
                   f"across {na['total_notices']} notices.")

    # 13) Business Insights ----------------------------------------------------- #
    if page == PAGES[12]:
        st.subheader("📊 Business Insights")
        st.caption("Apartment insights from the Phase-2 outputs — apartment_features "
                   "(current bed snapshot) and property_month. Per-apartment occupancy "
                   "is the CURRENT snapshot only (no historical apartment bookings). "
                   "Unavailable metrics show “Not Available”.")
        af = feats["apartment_features"].copy()
        bpm = feats["property_month"].sort_values("billing_period")
        bf = feats["bed_features"].copy()
        NA = "Not Available"

        c1, c2 = st.columns(2)
        # 1) Highest Current Occupancy Apartment — apartment_features (snapshot).
        if len(af):
            top = af.sort_values(["occupancy_pct", "occupied_beds"],
                                 ascending=False).iloc[0]
            c1.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>🏆 Highest Current "
                f"Occupancy Apartment</div><div class='kpi-value'>"
                f"{top['apartment_code']}</div><div class='kpi-label'>"
                f"{top['occupancy_pct']:.1f}% · {int(top['occupied_beds'])}/"
                f"{int(top['total_beds'])} beds (current snapshot)</div></div>",
                unsafe_allow_html=True)
        else:
            c1.markdown("<div class='kpi-card'><div class='kpi-label'>🏆 Highest "
                        "Current Occupancy Apartment</div><div class='kpi-value'>"
                        f"{NA}</div></div>", unsafe_allow_html=True)
        # 2) Peak Occupancy Month — property_month.
        if "occupancy_pct" in bpm.columns and bpm["occupancy_pct"].notna().any():
            pk = bpm.loc[bpm["occupancy_pct"].idxmax()]
            c2.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>📈 Peak Occupancy "
                f"Month</div><div class='kpi-value'>{str(pk['billing_period'])}</div>"
                f"<div class='kpi-label'>{pk['occupancy_pct']:.1f}% · "
                f"{int(pk['occupied_beds'])}/{config.TOTAL_BEDS} beds</div></div>",
                unsafe_allow_html=True)
        else:
            c2.markdown("<div class='kpi-card'><div class='kpi-label'>📈 Peak "
                        f"Occupancy Month</div><div class='kpi-value'>{NA}</div>"
                        "</div>", unsafe_allow_html=True)
        st.markdown("")

        # 3) Vacant Apartments / Beds — presentation sourced from live_bed_snapshot
        # so this section is IDENTICAL to the Available Beds dashboard's operational
        # vacancy. It intentionally does NOT use apartment_features/bed_features
        # vacancy here. No backend calculation, feature or health-score is changed;
        # apartment_features/bed_features/property_month remain untouched and are
        # still used by every other Business Insights metric above/below.
        live_snap = _live_bed_snapshot(cleaned["bookings"])
        _tenant_states = ["Occupied", "Notice", "Notice-Booked"]
        _apt = (live_snap.groupby("apartment_code", as_index=False)
                .agg(vacant_beds=("is_vacant", "sum"),
                     total_beds=("bed_id", "count"),
                     inactive_beds=("is_inactive", "sum"),
                     occupied_now=("live_status",
                                   lambda s: int(s.isin(_tenant_states).sum()))))
        _apt["operational_beds"] = _apt["total_beds"] - _apt["inactive_beds"]
        _apt["occupancy_pct"] = np.where(
            _apt["operational_beds"] > 0,
            (_apt["occupied_now"] / _apt["operational_beds"] * 100).round(1),
            0.0)

        st.markdown("#### 🏚️ Vacant Apartments / Beds")
        st.caption("Operational vacancy uses the same live bed classification as the "
                   "Available Beds dashboard (live_bed_snapshot), so both tabs match.")
        va = (_apt[_apt["vacant_beds"] > 0]
              .sort_values("vacant_beds", ascending=False)
              [["apartment_code", "vacant_beds", "occupancy_pct"]].copy())
        va["Status"] = "Vacant"
        va = va.rename(columns={"apartment_code": "Apartment",
                                "vacant_beds": "Vacant Beds",
                                "occupancy_pct": "Occupancy %"})
        if len(va):
            st.dataframe(va, use_container_width=True, height=240)
        else:
            st.success("No operational vacant beds right now.")

        # Inactive inventory shown separately (all-Not-Active beds, e.g. A22).
        inact = (_apt[_apt["inactive_beds"] > 0]
                 .sort_values("inactive_beds", ascending=False)
                 [["apartment_code", "inactive_beds"]].copy())
        if len(inact):
            inact["Status"] = "Inactive"
            inact = inact.rename(columns={"apartment_code": "Apartment",
                                          "inactive_beds": "Inactive Beds"})
            st.caption("Inactive inventory (Not-Active beds, excluded from "
                       "operational vacancy):")
            st.dataframe(inact, use_container_width=True, height=140)

        vbeds = (live_snap[live_snap["live_status"] == "Vacant"]
                 [["apartment_code", "bed_code", "live_status",
                   "current_rate"]].copy())
        if len(vbeds):
            vbeds["Estimated Vacant Duration"] = NA          # not in the data
            st.caption("Vacancy duration is not stored in the data — shown as "
                       f"“{NA}”. Currently vacant operational beds "
                       "(from live_bed_snapshot):")
            st.dataframe(vbeds.rename(columns={
                "apartment_code": "Apartment", "bed_code": "Bed Code",
                "live_status": "Current Status", "current_rate": "Monthly Rent"}),
                use_container_width=True, height=220)

        # 4) Rent Review Opportunity — apartment_features, occupancy >= 95%.
        st.markdown("#### 💰 Rent Review Opportunity")
        high = af[af["occupancy_pct"] >= 95].sort_values(
            "occupancy_pct", ascending=False).copy()
        if len(high):
            high["Current Status"] = np.where(high["occupancy_pct"] >= 100,
                                              "Fully Occupied", "Highly Occupied")
            st.dataframe(high[["apartment_code", "occupancy_pct", "Current Status"]]
                         .rename(columns={"apartment_code": "Apartment",
                                          "occupancy_pct": "Occupancy %"}),
                         use_container_width=True, height=240)
            st.caption("Apartment is currently highly occupied. Monitor future "
                       "occupancy before considering a rent revision.")
        else:
            st.info(f"No apartment is currently ≥95% occupied — {NA}.")

        # 5) Low Occupancy Apartments — apartment_features (lowest occupancy).
        st.markdown("#### 📉 Low Occupancy Apartments")
        low = (af.sort_values("occupancy_pct", ascending=True).head(10)
               [["apartment_code", "occupied_beds", "total_beds", "occupancy_pct"]]
               .rename(columns={"apartment_code": "Apartment",
                                "occupied_beds": "Occupied Beds",
                                "total_beds": "Total Beds",
                                "occupancy_pct": "Occupancy %"}))
        st.dataframe(low, use_container_width=True, height=300)


# --------------------------------------------------------------------------- #
# Static HTML fallback (no Streamlit needed)
# --------------------------------------------------------------------------- #
def export_html():
    cleaned, feats = _data()
    kpis = _kpis(cleaned, feats)
    figs = _figures(cleaned, feats)
    kpi_html = "".join(
        f"<div style='display:inline-block;margin:8px;padding:14px 20px;"
        f"border:1px solid #ddd;border-radius:10px;font-family:sans-serif'>"
        f"<div style='color:#666;font-size:13px'>{k}</div>"
        f"<div style='font-size:22px;font-weight:700'>{v}</div></div>"
        for k, v in kpis.items())
    parts = [f"<h1 style='font-family:sans-serif'>🏢 {APP_TITLE}</h1>", kpi_html]
    for fig in figs.values():
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    apt = _load_csv("forecast_apartment_summary.csv")
    if apt is not None:
        fig = px.bar(apt.head(15), x="apartment_code", y="next_month_amount",
                     title="Apartment-wise next-month electricity (top 15)")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    out = config.OUT_DIR / "dashboard.html"
    out.write_text("<html><body>" + "".join(parts) + "</body></html>",
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _sr
        if _sr.get_script_run_ctx() is not None:
            run_streamlit()
        else:
            export_html()
    except Exception:
        export_html()
