"""Vista Heights - PG Management Analytics dashboard.

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
    mv = _load_meta_json("model_meta_multivariate.json")
    comp = _load_csv("comparison_multivariate.csv")
    if mv is None or comp is None:
        st.info("Run: python -m src.revenue_multivariate")
        return
    ro, mo = mv["revenue_only_best"], mv["multivariate_best"]
    pm = feats["property_month"]
    mae = mo["mae"]                                   # for the 95% band

    # ---- Forecast KPI cards (same layout as the revenue-only section) ------ #
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast model", mo["model"])
    m2.metric("Walk-forward MAPE", f"{mo['mape']:.2f}%",
              delta=f"Improved by {ro['mape']-mo['mape']:.2f} MAPE points",
              delta_color="normal")
    m3.metric("Walk-forward MAE", f"₹{mae/1e5:.2f} L")
    m4.metric("Predicted Next Month",
              f"₹{mv.get('next_month_revenue', float('nan'))/1e5:.2f} L")

    # ---- Revenue forecast with 95% confidence band (multivariate) ---------- #
    fig = _forecast_fig(pm, "forecast_multivariate.csv", "revenue",
                        "Revenue + Occupancy forecast with 95% confidence band", mae)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- Walk-forward validation: actual vs multivariate predicted --------- #
    bt = _load_csv("backtest_multivariate.csv")
    if bt is not None:
        wf = go.Figure()
        wf.add_scatter(x=bt.billing_period, y=bt.actual, name="actual",
                       mode="lines+markers", line=dict(color=C_PRIMARY, width=3))
        wf.add_scatter(x=bt.billing_period, y=bt.multivariate, name="predicted",
                       mode="lines+markers", line=dict(dash="dot", color=C_ACCENT))
        wf.update_layout(title="Prediction vs Actual — walk-forward "
                               "(Revenue + Occupancy model)", hovermode="x unified",
                         margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(wf, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- Comparison overlay: actual / revenue-only / revenue+occupancy ----- #
    if bt is not None:
        f = go.Figure()
        f.add_scatter(x=bt.billing_period, y=bt.actual, name="actual",
                      mode="lines+markers", line=dict(color=C_PRIMARY, width=3))
        f.add_scatter(x=bt.billing_period, y=bt.revenue_only, name="Revenue-only",
                      mode="lines+markers", line=dict(dash="dash", color=C_RISK))
        f.add_scatter(x=bt.billing_period, y=bt.multivariate,
                      name="Revenue+Occupancy", mode="lines+markers",
                      line=dict(dash="dot", color=C_ACCENT))
        f.update_layout(title="Does occupancy help? Actual vs both models "
                              "(walk-forward)", hovermode="x unified",
                        margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)

    # ---- Secondary metrics + model comparison table ------------------------ #
    s1, s2 = st.columns(2)
    s1.metric(
    "Multivariate R²",
    f"{mo['r2']:.2f}",
    delta=f"Improved by {mo['r2']-ro['r2']:.2f}"
)
    s2.metric("Occupancy↔Revenue r", f"{mv['occupancy_revenue_corr']:.3f}")
    st.markdown("**Model comparison — Revenue-only vs Revenue+Occupancy** "
                "(identical walk-forward windows)")
    st.dataframe(comp.style.highlight_min(subset=["MAPE", "MAE", "RMSE"],
                                          color="rgba(42,157,143,.25)")
                 .highlight_max(subset=["R2"], color="rgba(42,157,143,.25)"),
                 use_container_width=True)
    st.markdown(_multivariate_verdict(mv))

    # Revenue vs Occupancy: correlation scatter + monthly trend.
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

    # Feature importance: permutation + SHAP.
    c1, c2 = st.columns(2)
    pi = _load_csv("perm_importance_multivariate.csv")
    if pi is not None:
        pi.columns = ["feature", "importance"]
        f = px.bar(pi.head(12).sort_values("importance"), x="importance",
                   y="feature", orientation="h",
                   title="Permutation importance (multivariate)",
                   color_discrete_sequence=[C_ACCENT])
        f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
    sh = _load_csv("shap_multivariate.csv")
    if sh is not None:
        sh.columns = ["feature", "mean_abs_shap"]
        f = px.bar(sh.head(12).sort_values("mean_abs_shap"), x="mean_abs_shap",
                   y="feature", orientation="h",
                   title="SHAP mean |value| (tree model)",
                   color_discrete_sequence=[C_PRIMARY])
        f.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c2.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)

    # Scenario analysis (±5% occupancy).
    st.markdown("**Scenario analysis — next-month revenue vs occupancy** "
                "(multivariate model, real next-month lag inputs)")
    sc_data = mv["scenario"]
    s1, s2, s3 = st.columns(3)
    s2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Base (current "
                f"occupancy)</div><div class='kpi-value'>₹{sc_data['base']/1e5:.2f} L"
                f"</div></div>", unsafe_allow_html=True)
    up = sc_data["plus_5pct"] - sc_data["base"]
    dn = sc_data["minus_5pct"] - sc_data["base"]
    s1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Occupancy −5%</div>"
                f"<div class='kpi-value' style='color:#C0392B'>"
                f"₹{sc_data['minus_5pct']/1e5:.2f} L</div>"
                f"<div class='kpi-label'>{dn/1e5:+.2f} L</div></div>",
                unsafe_allow_html=True)
    s3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Occupancy +5%</div>"
                f"<div class='kpi-value' style='color:#27AE60'>"
                f"₹{sc_data['plus_5pct']/1e5:.2f} L</div>"
                f"<div class='kpi-label'>{up/1e5:+.2f} L</div></div>",
                unsafe_allow_html=True)
    st.download_button("⬇️ Export multivariate comparison CSV",
                       comp.to_csv(index=False), "comparison_multivariate.csv",
                       "text/csv")
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
    inv = cleaned["invoices"]
    pm = feats["property_month"]
    mv = _load_meta_json("model_meta_multivariate.json")
    latest = inv["billing_period"].max()
    cur_rev = inv.loc[inv.billing_period == latest, "total_amount"].sum()
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
    mv = _load_meta_json("model_meta_multivariate.json")
    if mv:
        items.append(("🔗", "Occupancy↔Revenue Corr",
                      f"{mv['occupancy_revenue_corr']:.2f}"))
    return items


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _kpis(cleaned, feats):
    inv = cleaned["invoices"]
    beds = feats["bed_features"]
    ex = pd.to_datetime(cleaned["notices"]["estimated_exit_date"], utc=True,
                        errors="coerce")
    active_notices = int((ex >= pd.Timestamp.now(tz="UTC")).sum())
    return {
        "💰 Total Revenue": f"₹{inv['total_amount'].sum()/1e7:.2f} Cr",
        "🛏️ Occupancy": f"{_current_occupancy_pct(feats):.1f}%",
        "🚪 Vacant Beds": f"{int(beds['is_vacant'].sum())}",
        "📋 Beds on Notice": f"{int(beds['on_notice'].sum())}",
        "📤 Active Notices": f"{active_notices}",
        "⚡ Electricity Cost": f"₹{cleaned['electricity']['amount'].sum()/1e5:.0f} L",
        "🔧 Open Tickets": f"{int((cleaned['tickets']['status'] != 'closed').sum())}",
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
    figs["electricity"] = px.line(pm, x="month", y="elec_cost", markers=True,
                                  title="Electricity Cost",
                                  color_discrete_sequence=[C_WARN])
    bl = feats["bed_features"]["bed_lifecycle_status"].value_counts().reset_index()
    bl.columns = ["status", "count"]
    figs["beds"] = px.pie(bl, names="status", values="count",
                          title="Bed Lifecycle Mix", hole=0.45,
                          color="status",
                          color_discrete_map={"occupied": C_PRIMARY,
                                              "notice": C_RISK,
                                              "vacant": C_WARN,
                                              "booked": C_ACCENT})
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

    tabs = st.tabs([
        "📊 Executive Summary", "📈 Revenue Forecast", "🛏️ Occupancy Forecast",
        "🏠 Apartment-wise Forecast", "💰 Financial Overview",
        "👥 Tenant Segmentation", "💡 AI Recommendations",
        "📦 Asset Management", "🚪 Available Beds", "🔧 Maintenance",
        "🏆 Apartment Performance", "📤 Notice & Exit", "📊 Business Insights"])

    # 1) Executive Summary ---------------------------------------------------- #
    with tabs[0]:
        kpis = _kpis(cleaned, feats)
        cols = st.columns(len(kpis))
        for col, (k, v) in zip(cols, kpis.items()):
            col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{k}</div>"
                         f"<div class='kpi-value'>{v}</div></div>",
                         unsafe_allow_html=True)
        st.markdown("")

        c1, c2 = st.columns(2)
        c1.plotly_chart(figs["monthly_trend"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["electricity"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c1.plotly_chart(figs["beds"], use_container_width=True,
                        config=PLOTLY_CONFIG)
        c2.plotly_chart(figs["tickets"], use_container_width=True,
                        config=PLOTLY_CONFIG)

    # 2) Revenue Forecast ------------------------------------------------------ #
    with tabs[1]:
       
        # ---- Third model: multivariate revenue + occupancy ----------------- #
        st.markdown("## Revenue Forecast")
        st.markdown("### Revenue Forecast based on Historical Occupancy")
        st.caption("Revenue prediction is based on historical revenue and historical occupancy "
        "features from the real booking/stay dataset. Only lagged occupancy values "
        "are used to avoid data leakage.")
        _render_multivariate(st, feats)

    # 3) Occupancy Forecast ---------------------------------------------------- #
    with tabs[2]:
        st.subheader("Occupancy Forecast (Booking-based)")
        st.caption("Forecast based on real booking history (occupied beds). "
                    "Occupancy % = Occupied Beds / Total Beds (192).")

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
                    c3.metric("Next month Occupancy",
                              f"{int(occ.iloc[0]['occupied_beds'])} Beds "
                              f"({occ.iloc[0]['occupancy_pct']:.1f}%)")
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

        # ---- Additional ML cross-check vs Holt-Winters (guarded) ----------- #
        # Purely additive: only renders if src/occupancy_ml.py outputs exist.
        # Holt-Winters above stays the primary production forecast.
        ocmp = _load_csv("occupancy_model_comparison.csv")
        ometa = _load_meta_json("occupancy_model_metadata.json")
        if ocmp is not None and ometa is not None:
            st.markdown("---")
            st.markdown("### ML cross-check vs Holt-Winters (additional model)")
            st.caption("Second occupancy model: leakage-safe supervised ML on real "
                       "monthly property features, walk-forward validated on the same "
                       "months. Holt-Winters remains the primary forecaster.")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Primary (production)", "Holt-Winters",
                      f"MAPE {ometa['hw_mape']:.2f}%")
            g2.metric(f"Best ML ({ometa['ml_model']})", f"MAPE {ometa['ml_mape']:.2f}%",
                      delta=f"{ometa['improvement_mape_pct_points']:+.2f} vs HW",
                      delta_color="normal")
            g3.metric("Overall winner", ometa["winner"])
            g4.metric("Next month (ML)",
                      f"{ometa['next_month_occupied_beds']} Beds "
                      f"({ometa['next_month_occupancy_pct']:.1f}%)",
                      help=f"95% range: {ometa['next_month_lower_beds']}–"
                           f"{ometa['next_month_upper_beds']} beds")
            st.markdown(f"**ML forecast — {ometa['next_month_period']}: "
                        f"{ometa['next_month_occupied_beds']} beds "
                        f"({ometa['next_month_occupancy_pct']:.1f}%)** &nbsp;·&nbsp; "
                        f"95% range **{ometa['next_month_lower_beds']}–"
                        f"{ometa['next_month_upper_beds']} beds**.")
            st.dataframe(
                ocmp.set_index("Model")[["MAE", "RMSE", "MAPE", "R2"]]
                    .style.highlight_min(subset=["MAE", "RMSE", "MAPE"],
                                         color="rgba(42,157,143,.25)"),
                use_container_width=True)
            obt = _load_csv("occupancy_backtest_ml.csv")
            if obt is not None:
                wf = go.Figure()
                wf.add_scatter(x=obt.billing_period, y=obt.actual, name="actual",
                               mode="lines+markers",
                               line=dict(color=C_PRIMARY, width=3))
                wf.add_scatter(x=obt.billing_period, y=obt.hw_predicted,
                               name="Holt-Winters", mode="lines+markers",
                               line=dict(dash="dash", color=C_RISK))
                wf.add_scatter(x=obt.billing_period, y=obt.ml_predicted,
                               name=f"ML ({ometa['ml_model']})", mode="lines+markers",
                               line=dict(dash="dot", color=C_ACCENT))
                wf.update_layout(title="Occupied beds — actual vs Holt-Winters vs ML "
                                       "(walk-forward)", hovermode="x unified",
                                 margin=dict(l=10, r=10, t=48, b=10))
                st.plotly_chart(wf, use_container_width=True, config=PLOTLY_CONFIG)
            oml = _load_csv("forecast_occupancy_ml.csv")
            if oml is not None:
                st.download_button("⬇️ Export ML occupancy forecast (with 95% band) CSV",
                                   oml.to_csv(index=False),
                                   "forecast_occupancy_ml.csv", "text/csv")
            st.caption(ometa.get("capacity_note", ""))

    # 4) Apartment-wise Forecast ----------------------------------------------- #
    with tabs[3]:
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
    with tabs[4]:
        st.subheader("Financial Overview")
        st.caption("Revenue analytics from real invoice data only — billed revenue "
                   "= sum of invoice totals. No collection or payment figures: the "
                   "dataset has no real payments/receipts table, so nothing here is "
                   "inferred.")
        fyr = ops.financial_year_revenue(cleaned["invoices"])
        byfy = ops.revenue_by_financial_year(cleaned["invoices"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Financial Year", f"FY {fyr['fy_label']}")
        c2.metric("FY Revenue (billed)", f"₹{fyr['revenue']/1e5:.2f} L")
        c3.metric("Total Revenue (all-time)",
                  f"₹{cleaned['invoices']['total_amount'].sum()/1e7:.2f} Cr")

        f1 = px.bar(fyr["monthly"], x="month", y="revenue",
                    title=f"Financial Year Revenue Trend — FY {fyr['fy_label']}",
                    color_discrete_sequence=[C_PRIMARY])
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)

        byfy_lbl = byfy.assign(FY=byfy["fy"] + byfy["in_progress"].map(
            {True: " (in progress)", False: ""}))
        f3 = px.bar(byfy_lbl, x="FY", y="revenue",
                    title="Revenue by Financial Year (billed)",
                    color="in_progress",
                    color_discrete_map={False: C_PRIMARY, True: C_WARN})
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10), showlegend=False)
        st.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("The latest financial year is still in progress (fewer than 12 "
                   "months billed), so its bar is not directly comparable to "
                   "completed years. Financial year auto-detected (Apr–Mar); updates "
                   "as new invoice data is added.")

        st.dataframe(byfy[["fy", "revenue", "months", "invoices", "in_progress"]]
                     .rename(columns={"fy": "Financial Year", "revenue": "Revenue (₹)",
                                      "months": "Months billed",
                                      "invoices": "Invoices",
                                      "in_progress": "In progress"}),
                     use_container_width=True)
        st.download_button("⬇️ Export FY revenue summary CSV",
                           byfy.to_csv(index=False),
                           "revenue_by_financial_year.csv", "text/csv")

    # 6) Tenant Segmentation --------------------------------------------------- #
    with tabs[5]:
        st.subheader("Tenant segments — real billing behaviour")
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
    with tabs[6]:
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
                    f"Status : {r['status']}<br>"
                    f"Units Consumed : {int(round(float(r['units_consumed'])))}<br>"
                    f"Expected Usage : {int(r['expected_units'])}</div>",
                    unsafe_allow_html=True)
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

        # Transparency: the live page metrics this summary is built from.
        with st.expander("📊 Live dashboard metrics behind this summary"):
            fin = outputs["financial"]["fy"]
            occ = outputs["occupancy"]
            rf = outputs["revenue_forecast"]
            nt = outputs["notices"]
            ms = outputs["maintenance"]
            bd = outputs["beds"]
            src = {
                "Financial Overview": f"FY {fin['fy_label']} revenue "
                                      f"₹{fin['revenue']/1e5:.1f} L",
                "Revenue Forecast": (f"next month ₹{rf['next_month_revenue']/1e5:.1f} L"
                                     if rf else "n/a"),
                "Occupancy": (f"current {occ['current_pct']:.1f}%"
                              if occ["current_pct"] is not None else "n/a"),
                "Notice & Exit": f"{int(nt.get('upcoming_exits', 0))} upcoming exits",
                "Maintenance": f"{int(ms.get('open', 0))} open tickets",
                "Available Beds": f"{int(bd.get('vacant_beds', 0))} vacant beds",
            }
            st.table(pd.DataFrame({"Dashboard page": list(src),
                                   "Latest metric": list(src.values())}))

    # 8) Asset Management ------------------------------------------------------ #
    with tabs[7]:
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
    with tabs[8]:
        st.subheader("Available Beds")
        ba = ops.bed_availability(cleaned["beds_snapshot"])
        _kpi_cards(st, [
            ("🛏️ Total Beds", f"{ba['total_beds']}"),
            ("🚪 Vacant Beds", f"{ba['vacant_beds']}"),
            ("📊 Occupancy", f"{ba['occupancy_pct']}%"),
            ("💰 Vacant Revenue Opportunity",
             f"₹{ba['vacant_revenue_opportunity']/1e5:.2f} L/mo"),
        ])
        st.caption("Floor-wise vacancy is not available (no floor column); vacancy "
                   "is shown **block-wise** from the apartment-code prefix. Historical "
                   "vacancy trend is not possible — beds are a current snapshot.")
        c1, c2 = st.columns(2)
        f1 = px.pie(ba["lifecycle"], names="status", values="count", hole=0.45,
                    title="Bed Lifecycle Mix",
                    color="status", color_discrete_map={"occupied": C_PRIMARY,
                    "notice": C_RISK, "vacant": C_WARN, "booked": C_ACCENT})
        f2 = px.bar(ba["by_block"], x="block", y="vacancy_pct",
                    title="Vacancy % by Block", color="vacancy_pct",
                    color_continuous_scale="OrRd")
        f1.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        f2.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        c1.plotly_chart(f1, use_container_width=True, config=PLOTLY_CONFIG)
        c2.plotly_chart(f2, use_container_width=True, config=PLOTLY_CONFIG)
        apt_v = ba["by_apartment"]
        apt_v = apt_v[apt_v["vacant"] > 0]
        f3 = px.bar(apt_v, x="apartment_code", y="vacant",
                    title="Vacant Beds by Apartment",
                    color_discrete_sequence=[C_RISK])
        f3.update_layout(margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(f3, use_container_width=True, config=PLOTLY_CONFIG)
        blocks = st.multiselect("Filter block",
                                sorted(ba["by_block"]["block"].dropna().unique()))
        vt = ba["vacant_table"].copy()
        if blocks:
            vt = vt[vt["apartment_code"].str[0].isin(blocks)]
        st.dataframe(vt, use_container_width=True, height=300)
        st.download_button("⬇️ Export vacant beds CSV", vt.to_csv(index=False),
                           "vacant_beds.csv", "text/csv")
        st.caption(f"**Summary:** {ba['vacant_beds']} of {ba['total_beds']} beds "
                   f"vacant ({100-ba['occupancy_pct']:.1f}%), a monthly revenue "
                   f"opportunity of ₹{ba['vacant_revenue_opportunity']/1e5:.2f} L "
                   f"if filled at current rates.")

    # 10) Maintenance Performance ----------------------------------------------- #
    with tabs[9]:
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
    with tabs[10]:
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
    with tabs[11]:
        st.subheader("Notice & Exit Analytics")
        na = ops.notice_analytics(cleaned["notices"])
        _kpi_cards(st, [
            ("📋 Total Notices", f"{na['total_notices']}"),
            ("🚪 Upcoming Exits", f"{na['upcoming_exits']}"),
            ("💸 Monthly Revenue Impact",
             f"₹{na['monthly_revenue_impact']/1e5:.2f} L"),
            ("📅 Avg Notice Period", f"{na['avg_notice_days']:.0f} days"),
        ])
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
    with tabs[12]:
        st.subheader("📊 Business Insights")
        st.caption("Actionable insights from real data only. Per-apartment occupancy "
                   "is the current bed snapshot (the dataset has no per-apartment "
                   "occupancy history); peak-month occupancy uses the real monthly "
                   "booking series. Nothing is estimated — unavailable metrics show "
                   "“Not Available”.")
        bi = ops.business_insights(cleaned, feats)

        c1, c2 = st.columns(2)
        mb = bi["most_booked"]
        if mb:
            c1.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>🏆 Highest Current "
                f"Occupancy Apartment</div><div class='kpi-value'>{mb['apartment']}"
                f"</div>"
                f"<div class='kpi-label'>Block {mb['block']} · "
                f"{mb['occupancy_pct']:.1f}% occupancy · {mb['active_beds']} active "
                f"beds</div></div>", unsafe_allow_html=True)
        else:
            c1.markdown("<div class='kpi-card'><div class='kpi-label'>🏆 Highest "
                        "Current Occupancy Apartment</div><div class='kpi-value'>Not "
                        "Available</div></div>", unsafe_allow_html=True)
        pk = bi["peak_month"]
        if pk:
            c2.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>📈 Peak Occupancy "
                f"Month</div><div class='kpi-value'>{pk['month']}</div>"
                f"<div class='kpi-label'>{pk['occupancy_pct']:.1f}% · "
                f"{pk['occupied_beds']}/{pk['total_beds']} beds</div></div>",
                unsafe_allow_html=True)
        else:
            c2.markdown("<div class='kpi-card'><div class='kpi-label'>📈 Peak "
                        "Occupancy Month</div><div class='kpi-value'>Not Available"
                        "</div></div>", unsafe_allow_html=True)
        st.markdown("")

        # 🏚️ Longest vacant rooms/beds — duration not in data, list vacant beds.
        st.markdown("#### 🏚️ Vacant Rooms / Beds")
        vb = bi["vacant_beds"]
        if len(vb):
            st.caption("Vacancy duration is not stored in the dataset (beds are a "
                       "current snapshot), so it shows “Not Available”. Listing the "
                       "currently vacant beds instead.")
            st.dataframe(vb, use_container_width=True, height=280)
        else:
            st.success("No vacant beds right now.")

        # 💰 Rent increase opportunity — recommendation only, no rent calculation.
        st.markdown("#### 💰 Rent Increase Opportunity")
        ro = bi["rent_opportunity"]
        if len(ro):
            st.dataframe(ro, use_container_width=True, height=240)
        else:
            st.info("No apartment is currently above 95% occupancy — Not Available.")

        # 📉 Low demand apartments — most vacant beds.
        st.markdown("#### 📉 Low Demand Apartments")
        ld = bi["low_demand"]
        if len(ld):
            st.dataframe(ld, use_container_width=True, height=280)
        else:
            st.success("No apartment has vacant beds right now.")


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
