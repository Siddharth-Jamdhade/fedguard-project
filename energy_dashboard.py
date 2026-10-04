"""
energy_dashboard.py — PS-13 FedGuard Streamlit Dashboard

Addresses ALL PS-13 requirements:
  ✓ Records energy consumption (power → kWh per interval)
  ✓ Estimates cost (₹/kWh) and CO₂ (kg) per node and combined
  ✓ Compares usage across 3 spaces AND across first/second half (time periods)
  ✓ Detects abnormal usage automatically (rolling-window ML + MCSA)
  ✓ Explains the likely cause (MCSA: power vs vibration pattern)
  ✓ Suggests specific actions with estimated monthly savings (₹ + CO₂ kg)

Run:
    streamlit run energy_dashboard.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from energy_carbon import (
    summarize_period, compare_periods, project_monthly,
    TARIFF_INR_PER_KWH, EMISSION_FACTOR_KG_CO2_PER_KWH,
)
from energy_anomaly import diagnose

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedGuard — PS-13 Energy Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#06090F;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.2rem;max-width:1400px;}
[data-testid="stSidebar"]{background:rgba(14,20,36,0.95);border-right:1px solid rgba(120,150,210,0.15);}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{color:#8899bb;font-size:0.8rem;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#0c1120;border-color:rgba(120,150,210,0.2);color:#e8edf8;}
.mono{font-family:'IBM Plex Mono',monospace;}
.top-bar{display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:16px;margin-bottom:20px;border-bottom:1px solid rgba(120,150,210,0.18);}
.top-title{font-size:1.5rem;font-weight:700;color:#e8edf8;letter-spacing:-0.01em;}
.top-title .accent{color:#F2C879;}
.top-sub{font-family:'IBM Plex Mono',monospace;font-size:0.73rem;color:#6a7a9a;margin-top:4px;letter-spacing:0.04em;}
.status-pill{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;font-weight:600;color:#54D6A0;background:rgba(84,214,160,0.1);border:1px solid rgba(84,214,160,0.3);padding:5px 14px;border-radius:3px;letter-spacing:0.06em;}
.kpi-card{background:rgba(18,26,48,0.7);border:1px solid rgba(120,150,210,0.15);border-radius:8px;padding:16px 20px;margin:2px;}
.kpi-label{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#6a7a9a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;}
.kpi-value{font-size:1.6rem;font-weight:700;color:#e8edf8;line-height:1.1;}
.kpi-sub{font-size:0.75rem;color:#8899bb;margin-top:2px;}
.anomaly-high{background:rgba(255,122,107,0.08);border-left:3px solid #FF7A6B;border-radius:6px;padding:14px 18px;margin:8px 0;}
.anomaly-medium{background:rgba(242,200,121,0.08);border-left:3px solid #F2C879;border-radius:6px;padding:14px 18px;margin:8px 0;}
.anomaly-low{background:rgba(111,214,232,0.08);border-left:3px solid #6FD6E8;border-radius:6px;padding:14px 18px;margin:8px 0;}
.section-label{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;font-weight:600;color:#F2C879;text-transform:uppercase;letter-spacing:0.12em;margin:24px 0 8px 0;}
.privacy-badge{background:rgba(84,214,160,0.07);border:1px solid rgba(84,214,160,0.25);border-radius:6px;padding:10px 16px;font-size:0.78rem;color:#54D6A0;margin-top:8px;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
NODES      = ["A", "B", "C"]
NODE_NAMES = {
    "A": "Business A — Factory",
    "B": "Business B — Office",
    "C": "Business C — Workshop",
}
NODE_COLORS = {"A": "#F2C879", "B": "#6FD6E8", "C": "#54D6A0"}
WINDOW_SIZE = 5


def hex_rgba(h: str, a: float = 0.15) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#8899bb"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=36, b=10),
)


# ─── Data helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_csv(node: str) -> pd.DataFrame | None:
    path = f"energy_{node}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    df["power_w"] = pd.to_numeric(df["power_w"], errors="coerce")
    if "vibration_rms" not in df.columns:
        if all(c in df.columns for c in ["ax", "ay", "az"]):
            df["vibration_rms"] = np.sqrt(
                df["ax"].astype(float)**2 +
                df["ay"].astype(float)**2 +
                df["az"].astype(float)**2)
        else:
            df["vibration_rms"] = 9.81
    df["vibration_rms"] = pd.to_numeric(df["vibration_rms"], errors="coerce")
    df = df.dropna(subset=["power_w", "vibration_rms"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=120)
def load_global_model() -> dict | None:
    if os.path.exists("global_energy_model.pkl"):
        with open("global_energy_model.pkl", "rb") as f:
            return pickle.load(f)
    return None


def compute_metrics(df: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    df = df.copy()
    hours = interval_min / 60.0
    df["energy_kwh"]  = df["power_w"] * hours / 1000.0
    df["cost_inr"]    = df["energy_kwh"] * TARIFF_INR_PER_KWH
    df["co2_kg"]      = df["energy_kwh"] * EMISSION_FACTOR_KG_CO2_PER_KWH
    df["cum_energy"]  = df["energy_kwh"].cumsum()
    df["cum_cost"]    = df["cost_inr"].cumsum()
    df["cum_co2"]     = df["co2_kg"].cumsum()
    return df


def detect_anomalies(df: pd.DataFrame, node: str) -> list:
    bn = max(1, int(len(df) * 0.30))
    bp = float(df["power_w"].iloc[:bn].mean())
    bv = float(df["vibration_rms"].iloc[:bn].mean())
    events = []
    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w   = df.iloc[i:i + WINDOW_SIZE]
        ap  = float(w["power_w"].mean())
        av  = float(w["vibration_rms"].mean())
        d   = diagnose(ap, bp, av, bv)
        if d["severity"] != "none":
            d.update({"window_start": i, "node": node,
                      "avg_power_w": ap, "avg_vib": av})
            events.append(d)
    return events, bp, bv


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    sel_nodes = st.multiselect("Active Nodes", NODES, default=NODES)
    interval_min = st.selectbox(
        "Reading Interval",
        [1/60, 1.0, 5.0, 15.0],
        index=3,
        format_func=lambda x: {1/60: "1 second (Wokwi live)",
                                1.0:  "1 minute",
                                5.0:  "5 minutes",
                                15.0: "15 minutes (demo CSV)"}[x],
    )
    budget_kwh = st.number_input("Monthly Energy Budget (kWh)", value=1500, step=100)
    tariff     = st.number_input("Tariff (₹ / kWh)", value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("### 🌐 Federation")
    gm = load_global_model()
    if gm:
        st.success(f"✓ Global model loaded\nNodes: {', '.join(gm['contributing_nodes'])}\nSamples: {gm['total_samples']}")
    else:
        st.warning("No global model.\nRun:\n```\npython energy_federated_server.py --nodes weights_energy_A.pkl weights_energy_B.pkl weights_energy_C.pkl\n```")

    st.markdown("""
    <div class="privacy-badge">
    🔒 Raw sensor data stays local.<br>
    Only model weights (4 numbers per node) travel to the aggregator.
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div>
    <div class="top-title">⚡ FedGuard <span class="accent">Energy</span> — PS-13</div>
    <div class="top-sub">SMART ENERGY CONSUMPTION & CARBON FOOTPRINT MONITORING · FEDERATED MSME NETWORK</div>
  </div>
  <div class="status-pill">● SYSTEM ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────
all_df   : dict[str, pd.DataFrame] = {}
all_anom : dict[str, list]         = {}
baselines: dict[str, dict]         = {}
summaries: dict[str, dict]         = {}

for nd in sel_nodes:
    raw = load_csv(nd)
    if raw is None:
        st.warning(f"⚠ energy_{nd}.csv not found. Run `python generate_energy_data.py` first.", icon="⚠️")
        continue
    df = compute_metrics(raw, interval_min)
    all_df[nd] = df
    evts, bp, bv = detect_anomalies(df, nd)
    all_anom[nd] = evts
    baselines[nd] = {"power": bp, "vib": bv}
    summaries[nd]  = summarize_period(df["power_w"].tolist(), interval_min)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Energy Overview",
    "📊 Usage Comparison",
    "🚨 Anomaly Detection",
    "🌐 Federated Learning",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ENERGY OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not all_df:
        st.info("No data loaded. Generate demo CSVs first.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_kwh   = sum(s["energy_kwh"]  for s in summaries.values())
    total_cost  = total_kwh * tariff
    total_co2   = total_kwh * EMISSION_FACTOR_KG_CO2_PER_KWH
    total_anom  = sum(len(v) for v in all_anom.values())
    curr_power  = sum(all_df[n]["power_w"].iloc[-1] for n in all_df)
    proj        = project_monthly(curr_power / max(len(all_df), 1))

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, sub in [
        (c1, "⚡ Current Load",    f"{curr_power:.0f} W",    f"Proj. ₹{proj['projected_monthly_cost_inr']:.0f}/mo"),
        (c2, "🔋 Total Energy",    f"{total_kwh:.3f} kWh",   f"{len(all_df)} nodes combined"),
        (c3, "💰 Estimated Cost",  f"₹{total_cost:.2f}",     f"@ ₹{tariff}/kWh"),
        (c4, "🌱 CO₂ Emitted",    f"{total_co2:.3f} kg",     f"{EMISSION_FACTOR_KG_CO2_PER_KWH} kg/kWh"),
        (c5, "🚨 Anomalies",      str(total_anom),           "⚠ See Anomaly tab" if total_anom else "✓ All clear"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">▌ Power Consumption — All Nodes</div>',
                unsafe_allow_html=True)

    # ── Power timeline ────────────────────────────────────────────────────────
    fig_pw = go.Figure()
    for nd, df in all_df.items():
        col = NODE_COLORS[nd]
        fig_pw.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["power_w"],
            name=NODE_NAMES[nd],
            line=dict(color=col, width=2),
            fill="tozeroy",
            fillcolor=hex_rgba(col, 0.08),
        ))
    fig_pw.update_layout(**PLOTLY_LAYOUT, height=300,
                         xaxis_title="Reading #", yaxis_title="Power (W)")
    st.plotly_chart(fig_pw, use_container_width=True)

    # ── Cumulative cost + CO₂ gauge ──────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-label">▌ Cumulative Cost (₹)</div>',
                    unsafe_allow_html=True)
        fig_cost = go.Figure()
        for nd, df in all_df.items():
            col = NODE_COLORS[nd]
            fig_cost.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df["cum_cost"] * (tariff / TARIFF_INR_PER_KWH),
                name=NODE_NAMES[nd],
                line=dict(color=col, width=2),
                stackgroup="one",
                fillcolor=hex_rgba(col, 0.35),
            ))
        fig_cost.update_layout(**PLOTLY_LAYOUT, height=260,
                               xaxis_title="Reading #", yaxis_title="₹")
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-label">▌ CO₂ vs Monthly Budget</div>',
                    unsafe_allow_html=True)
        budget_co2 = budget_kwh * EMISSION_FACTOR_KG_CO2_PER_KWH
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(total_co2, 3),
            number={"suffix": " kg", "font": {"size": 28}},
            delta={"reference": budget_co2 * 0.10,
                   "relative": False,
                   "valueformat": ".3f",
                   "suffix": " kg"},
            title={"text": "CO₂ Emitted", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, budget_co2], "ticksuffix": " kg"},
                "bar": {"color": "#54D6A0"},
                "steps": [
                    {"range": [0,               budget_co2 * 0.5], "color": "#0d2018"},
                    {"range": [budget_co2 * 0.5, budget_co2 * 0.8], "color": "#2a1a08"},
                    {"range": [budget_co2 * 0.8, budget_co2],      "color": "#2a0a08"},
                ],
                "threshold": {
                    "line": {"color": "#FF7A6B", "width": 3},
                    "thickness": 0.75,
                    "value": budget_co2 * 0.8,
                },
            },
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Per-node summary table ────────────────────────────────────────────────
    st.markdown('<div class="section-label">▌ Per-Node Summary</div>',
                unsafe_allow_html=True)
    rows = []
    for nd in all_df:
        s = summaries[nd]
        p = project_monthly(s["avg_power_w"])
        rows.append({
            "Node":               NODE_NAMES[nd],
            "Avg Power (W)":      s["avg_power_w"],
            "Peak Power (W)":     s["peak_power_w"],
            "Energy (kWh)":       s["energy_kwh"],
            "Cost (₹)":           round(s["energy_kwh"] * tariff, 2),
            "CO₂ (kg)":           round(s["co2_kg"], 4),
            "Proj. Monthly ₹":    p["projected_monthly_cost_inr"],
            "Proj. Monthly CO₂":  p["projected_monthly_co2_kg"],
            "Anomaly Windows":    len(all_anom.get(nd, [])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — USAGE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not all_df:
        st.info("No data loaded.")
    else:
        st.markdown('<div class="section-label">▌ Energy / Cost / CO₂ — Across Spaces</div>',
                    unsafe_allow_html=True)

        nd_list = list(all_df.keys())
        names   = [NODE_NAMES[n] for n in nd_list]
        colors  = [NODE_COLORS[n] for n in nd_list]
        e_vals  = [summaries[n]["energy_kwh"]                       for n in nd_list]
        c_vals  = [summaries[n]["energy_kwh"] * tariff               for n in nd_list]
        co2_vals= [summaries[n]["energy_kwh"] * EMISSION_FACTOR_KG_CO2_PER_KWH for n in nd_list]

        fig_cmp = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Energy (kWh)", "Estimated Cost (₹)", "CO₂ Emitted (kg)"],
        )
        for col_idx, (yvals, fmt) in enumerate(
                [(e_vals, ".3f"), (c_vals, ".2f"), (co2_vals, ".4f")], 1):
            fig_cmp.add_trace(go.Bar(
                x=names, y=yvals, marker_color=colors,
                text=[f"{v:{fmt}}" for v in yvals],
                textposition="outside",
                showlegend=False,
            ), row=1, col=col_idx)
        fig_cmp.update_layout(**PLOTLY_LAYOUT, height=360,
                              title="Comparison Across Business Spaces")
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Time-period comparison ────────────────────────────────────────────
        st.markdown('<div class="section-label">▌ Time Period Comparison — First Half vs Second Half</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(all_df))
        for i, (nd, df) in enumerate(all_df.items()):
            mid  = len(df) // 2
            comp = compare_periods(
                df["power_w"].iloc[:mid].tolist(),
                df["power_w"].iloc[mid:].tolist(),
                interval_minutes=interval_min,
            )
            with cols[i]:
                st.markdown(f"**{NODE_NAMES[nd]}**")
                delta_e = comp["delta_energy_kwh"]
                dcolor  = "inverse" if delta_e > 0 else "normal"
                st.metric("Period 1 Energy", f"{comp['period_a']['energy_kwh']:.4f} kWh")
                st.metric("Period 2 Energy", f"{comp['period_b']['energy_kwh']:.4f} kWh",
                          delta=f"{delta_e:+.4f} kWh ({comp['pct_change']:+.1f}%)",
                          delta_color=dcolor)
                st.caption(f"Cost impact: ₹{comp['delta_cost_inr']:+.4f}")
                st.caption(f"CO₂ impact: {comp['delta_co2_kg']:+.4f} kg")

        # ── Stacked area: power timeline side by side ─────────────────────────
        st.markdown('<div class="section-label">▌ Power Profile — Overlay</div>',
                    unsafe_allow_html=True)
        fig_ov = go.Figure()
        for nd, df in all_df.items():
            fig_ov.add_trace(go.Scatter(
                x=list(range(len(df))), y=df["power_w"],
                name=NODE_NAMES[nd],
                line=dict(color=NODE_COLORS[nd], width=2),
            ))
        fig_ov.update_layout(**PLOTLY_LAYOUT, height=300,
                             xaxis_title="Reading #", yaxis_title="Power (W)")
        st.plotly_chart(fig_ov, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not all_df:
        st.info("No data loaded.")
    else:
        all_events = [(nd, e) for nd, evts in all_anom.items() for e in evts]
        total_ev   = len(all_events)
        high_ev    = sum(1 for _, e in all_events if e["severity"] == "high")
        med_ev     = sum(1 for _, e in all_events if e["severity"] == "medium")
        low_ev     = sum(1 for _, e in all_events if e["severity"] == "low")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Total Anomaly Windows", total_ev)
        with r2:
            st.metric("🔴 High Severity", high_ev)
        with r3:
            st.metric("🟡 Medium Severity", med_ev)
        with r4:
            st.metric("🔵 Low Severity", low_ev)

        if not all_events:
            st.success("✅ No anomalies detected across all active nodes.")
        else:
            # Sort: high first
            sev_order = {"high": 0, "medium": 1, "low": 2}
            sorted_events = sorted(all_events,
                                   key=lambda x: sev_order.get(x[1]["severity"], 3))

            st.markdown('<div class="section-label">▌ Detected Anomalies — Explained</div>',
                        unsafe_allow_html=True)

            for nd, ev in sorted_events:
                sev   = ev["severity"]
                css   = f"anomaly-{sev}"
                icon  = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(sev, "⚪")
                title = f"{icon} [{sev.upper()}] {NODE_NAMES[nd]} · Window {ev['window_start']}–{ev['window_start']+WINDOW_SIZE}"

                with st.expander(title, expanded=(sev == "high")):
                    st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
                    col_l, col_r = st.columns([3, 1])
                    with col_l:
                        st.markdown(f"**🔍 Likely Cause**\n\n{ev['likely_cause']}")
                        st.markdown(f"**💡 Suggested Action**\n\n{ev['suggested_action']}")
                    with col_r:
                        st.metric("Avg Power", f"{ev['avg_power_w']:.1f} W",
                                  delta=f"{ev['avg_power_w'] - baselines[nd]['power']:+.1f} W vs baseline",
                                  delta_color="inverse")
                        st.metric("Avg Vibration", f"{ev['avg_vib']:.4f}")
                        st.metric("Est. Monthly Savings",
                                  f"₹{ev['estimated_monthly_savings_inr']:.0f}")
                        st.metric("CO₂ Reduction / mo",
                                  f"{ev['estimated_monthly_co2_reduction_kg']:.2f} kg")
                    st.markdown("</div>", unsafe_allow_html=True)

        # ── Power vs Vibration scatter ──────────────────────────────────────
        st.markdown('<div class="section-label">▌ Power vs Vibration RMS — Anomaly Map</div>',
                    unsafe_allow_html=True)
        fig_sc = go.Figure()
        for nd, df in all_df.items():
            anomaly_windows = set(e["window_start"] for e in all_anom.get(nd, []))
            mask_norm = [True] * len(df)
            mask_anom = [False] * len(df)
            for ws in anomaly_windows:
                for j in range(ws, min(ws + WINDOW_SIZE, len(df))):
                    mask_norm[j] = False
                    mask_anom[j] = True
            mn = pd.Series(mask_norm)
            ma = pd.Series(mask_anom)
            c  = NODE_COLORS[nd]
            fig_sc.add_trace(go.Scatter(
                x=df["power_w"][mn], y=df["vibration_rms"][mn],
                mode="markers", name=f"{NODE_NAMES[nd]} — normal",
                marker=dict(color=c, size=4, opacity=0.55),
            ))
            if ma.any():
                fig_sc.add_trace(go.Scatter(
                    x=df["power_w"][ma], y=df["vibration_rms"][ma],
                    mode="markers", name=f"{NODE_NAMES[nd]} — anomaly",
                    marker=dict(color="#FF7A6B", size=7, symbol="x",
                                line=dict(width=1, color="#FF2200")),
                ))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=340,
                             xaxis_title="Power (W)",
                             yaxis_title="Vibration RMS",
                             title="Normal vs anomalous readings (× = anomaly window)")
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Anomaly savings summary ──────────────────────────────────────────
        if all_events:
            st.markdown('<div class="section-label">▌ Potential Monthly Savings if Anomalies Resolved</div>',
                        unsafe_allow_html=True)
            save_rows = []
            for nd, evts in all_anom.items():
                tot_save = sum(e["estimated_monthly_savings_inr"] for e in evts)
                tot_co2  = sum(e["estimated_monthly_co2_reduction_kg"] for e in evts)
                if tot_save > 0 or tot_co2 > 0:
                    save_rows.append({
                        "Node": NODE_NAMES[nd],
                        "Anomaly Windows": len(evts),
                        "Potential Monthly Savings (₹)": f"₹{tot_save:.0f}",
                        "CO₂ Reduction (kg/month)": f"{tot_co2:.2f}",
                    })
            if save_rows:
                st.dataframe(pd.DataFrame(save_rows), use_container_width=True,
                             hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEDERATED LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">▌ Node Training Status</div>',
                unsafe_allow_html=True)

    node_cols = st.columns(3)
    for i, nd in enumerate(NODES):
        pf = f"weights_energy_{nd}.pkl"
        with node_cols[i]:
            st.markdown(f"**{NODE_NAMES[nd]}**")
            if os.path.exists(pf):
                with open(pf, "rb") as f:
                    w = pickle.load(f)
                st.success(
                    f"✓ Trained\n\n"
                    f"Samples: {w['n_samples']}\n"
                    f"Accuracy: {w['accuracy']:.3f}\n"
                    f"Baseline power: {w['baseline_power_w']:.1f} W\n"
                    f"Baseline vib:   {w['baseline_vibration_rms']:.4f}"
                )
            else:
                st.error(f"✗ Not trained\n\nRun:\n`python energy_local_model.py --node {nd} --csv energy_{nd}.csv`")

    st.markdown('<div class="section-label">▌ Global Model</div>',
                unsafe_allow_html=True)

    if gm:
        g1, g2 = st.columns([1, 2])
        with g1:
            st.success(
                f"✓ Global model ready\n\n"
                f"Nodes:   {', '.join(gm['contributing_nodes'])}\n"
                f"Samples: {gm['total_samples']}"
            )
            if "node_baselines" in gm:
                st.markdown("**Node baselines in global model:**")
                bl_rows = [
                    {"Node": nd,
                     "Baseline Power (W)": f"{v['baseline_power_w']:.1f}",
                     "Baseline Vib":       f"{v['baseline_vibration_rms']:.4f}",
                     "Accuracy":           f"{v['accuracy']:.3f}"}
                    for nd, v in gm["node_baselines"].items()
                ]
                st.dataframe(pd.DataFrame(bl_rows), hide_index=True)
        with g2:
            st.markdown("#### Privacy-Preserving FL Architecture")
            st.code("""
┌────────────────────────────────────────────────────────────┐
│                 FEDGUARD PS-13 — HOW IT WORKS             │
│                                                            │
│  Business A        Business B        Business C            │
│  ┌─────────┐       ┌─────────┐       ┌─────────┐          │
│  │energy_A │       │energy_B │       │energy_C │  ← CSVs  │
│  │.csv     │       │.csv     │       │.csv     │  stay     │
│  │(LOCAL)  │       │(LOCAL)  │       │(LOCAL)  │  local    │
│  └────┬────┘       └────┬────┘       └────┬────┘          │
│       │ train           │ train           │ train          │
│       ▼                 ▼                 ▼                │
│  weights_A.pkl    weights_B.pkl    weights_C.pkl           │
│  (4 numbers)      (4 numbers)      (4 numbers)             │
│       │                 │                 │                │
│       └────────┬────────┘─────────────────┘               │
│                ▼                                           │
│         FedAvg aggregation                                 │
│         (energy_federated_server.py)                       │
│                │                                           │
│                ▼                                           │
│         global_energy_model.pkl  →  back to all nodes      │
└────────────────────────────────────────────────────────────┘
            """, language="text")
    else:
        st.info("No global model yet. Train all nodes first, then run the federated server.")

    st.markdown('<div class="section-label">▌ Run Commands</div>',
                unsafe_allow_html=True)
    st.code("""
# Step 1 — Generate demo data (or use Wokwi CSV output)
python generate_energy_data.py

# Step 2 — Train each node locally
python energy_local_model.py --node A --csv energy_A.csv
python energy_local_model.py --node B --csv energy_B.csv
python energy_local_model.py --node C --csv energy_C.csv

# Step 3 — Federated aggregation
python energy_federated_server.py \\
    --nodes weights_energy_A.pkl weights_energy_B.pkl weights_energy_C.pkl

# Step 4 — Launch dashboard
streamlit run energy_dashboard.py
    """, language="bash")
