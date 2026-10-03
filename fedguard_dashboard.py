"""
fedguard_dashboard.py — FedGuard dashboard
Control-room / industrial-monitoring visual design: this is built for MSME
equipment operators, not a generic AI demo, so the interface borrows from
real factory-floor instrumentation — graphite panels, amber caution accents,
monospace data readouts.

Run: streamlit run fedguard_dashboard.py
Works directly on the CSVs + .pkl files already in this folder
(synthetic today, real MPU6050 data later — same file names, zero code change).

Local-first: no email services, no cloud (ThingSpeak) uploads — everything
runs on this machine against the local CSVs, weights and audit DB.
"""

import pickle
import datetime
import numpy as np
import pandas as pd
import streamlit as st

from local_model import extract_features
from federated_server import load_global_model
from storage import log_federation_round, get_federation_history, get_local_history, get_dataset_loads

st.set_page_config(page_title="FedGuard — Federated Predictive Maintenance", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# THEME — control-room instrumentation palette
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-void: #070C17;
    --bg-panel: rgba(16, 24, 42, 0.75);
    --bg-panel-raised: rgba(23, 33, 55, 0.85);
    --border: rgba(122, 152, 207, 0.16);
    --border-bright: rgba(122, 152, 207, 0.34);
    --amber: #F2C879;
    --amber-dim: rgba(242, 200, 121, 0.16);
    --cyan: #6FD6E8;
    --safe: #54D6A0;
    --safe-dim: rgba(84, 214, 160, 0.10);
    --critical: #FF7A6B;
    --critical-dim: rgba(255, 122, 107, 0.10);
    --text-hi: #F1F4FA;
    --text-mid: #A5B2CC;
    --text-low: #5E6E8C;
    --gold-grad: linear-gradient(135deg, #F6D98A 0%, #E8B45A 45%, #C9983F 100%);
    --cyan-grad: linear-gradient(135deg, #8BE4F4 0%, #4FB8D4 100%);
    --panel-grad: linear-gradient(160deg, rgba(28, 40, 68, 0.85), rgba(12, 19, 34, 0.92));
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--bg-void); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

/* ---- Sidebar: dark to match the rest of the app ---- */
[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: var(--text-mid); font-size: 0.8rem; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--bg-void);
    border-color: var(--border);
    color: var(--text-hi);
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color: var(--text-hi); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--text-low); }

.mono { font-family: 'IBM Plex Mono', monospace; }

/* ---- Top bar ---- */
.fg-topbar {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding-bottom: 18px; margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
}
.fg-title { font-size: 1.55rem; font-weight: 700; color: var(--text-hi); letter-spacing: -0.01em; }
.fg-title .fg-lock { color: var(--amber); }
.fg-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text-mid);
    margin-top: 6px; letter-spacing: 0.02em; }
.fg-status-pill {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
    color: var(--safe); background: var(--safe-dim); border: 1px solid rgba(95,174,122,0.35);
    padding: 6px 14px; border-radius: 3px; letter-spacing: 0.05em; text-transform: uppercase;
}

/* ---- Section labels ---- */
.fg-section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
    color: var(--amber); text-transform: uppercase; letter-spacing: 0.12em;
    margin: 28px 0 4px 0; display: flex; align-items: center; gap: 8px;
}
.fg-section-label::before { content: ''; width: 3px; height: 14px; background: var(--amber); display: inline-block; }
.fg-section-sub { font-size: 0.85rem; color: var(--text-mid); margin-bottom: 14px; }

/* ---- Flow diagram (the signature element) ---- */
.fg-flow {
    display: flex; align-items: stretch; gap: 0; background: var(--bg-panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 20px 16px;
    margin-bottom: 8px; overflow-x: auto;
}
.fg-flow-node {
    flex: 1; min-width: 130px; text-align: center; padding: 0 10px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.fg-flow-node .icon { font-size: 1.4rem; margin-bottom: 6px; }
.fg-flow-node .label { font-size: 0.78rem; font-weight: 600; color: var(--text-hi); }
.fg-flow-node .sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text-low); margin-top: 3px; }
.fg-flow-node.local { background: rgba(95,174,122,0.07); border-radius: 4px; padding-top: 10px; padding-bottom: 10px; }
.fg-flow-arrow { display: flex; align-items: center; color: var(--border-bright); font-size: 1.1rem; padding: 0 4px; }

/* ---- Node cards ---- */
.fg-card {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 18px 18px 14px 18px; height: 100%; position: relative; overflow: hidden;
}
.fg-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--card-accent, var(--border-bright));
}
.fg-card-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.fg-card-title { font-size: 1.02rem; font-weight: 700; color: var(--text-hi); }
.fg-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; font-weight: 600;
    padding: 3px 8px; border-radius: 3px; letter-spacing: 0.04em; white-space: nowrap;
}
.fg-badge.hw { background: rgba(95,184,217,0.14); color: var(--cyan); border: 1px solid rgba(95,184,217,0.3); }
.fg-badge.sim { background: rgba(154,165,181,0.1); color: var(--text-mid); border: 1px solid var(--border-bright); }

.fg-stat-row { display: flex; gap: 22px; margin: 12px 0 14px 0; }
.fg-stat .val { font-family: 'IBM Plex Mono', monospace; font-size: 1.65rem; font-weight: 600; color: var(--text-hi); line-height: 1; }
.fg-stat .lbl { font-size: 0.68rem; color: var(--text-mid); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

.fg-lock-line {
    display: flex; align-items: center; gap: 7px; font-size: 0.75rem; color: var(--safe);
    background: var(--safe-dim); border: 1px solid rgba(95,174,122,0.25);
    padding: 7px 10px; border-radius: 4px; margin-bottom: 10px; font-weight: 500;
}
.fg-weights-label { font-size: 0.68rem; color: var(--text-low); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.fg-weights-box {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--cyan);
    background: var(--bg-void); border: 1px solid var(--border); border-radius: 4px;
    padding: 8px 10px; word-break: break-all; line-height: 1.5;
}

/* ---- Aggregation banner ---- */
.fg-agg-banner {
    background: linear-gradient(90deg, rgba(232,169,76,0.08), transparent);
    border: 1px solid var(--amber-dim); border-left: 3px solid var(--amber);
    border-radius: 4px; padding: 12px 16px; font-size: 0.85rem; color: var(--text-hi); margin-bottom: 18px;
}
.fg-agg-banner b { color: var(--amber); }

/* ---- Table ---- */
.fg-table { width: 100%; border-collapse: collapse; margin: 6px 0 16px 0; }
.fg-table th {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--text-mid); text-align: left; padding: 10px 14px;
    border-bottom: 1px solid var(--border-bright); font-weight: 600;
}
.fg-table td { padding: 12px 14px; border-bottom: 1px solid var(--border); font-size: 0.88rem; color: var(--text-hi); }
.fg-table td.mono-val { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.fg-table tr:last-child td { border-bottom: none; }

.fg-note {
    font-size: 0.82rem; color: var(--text-mid); background: var(--bg-panel);
    border: 1px solid var(--border); border-radius: 4px; padding: 12px 14px; line-height: 1.6;
}
.fg-note b { color: var(--cyan); }

/* ---- Live status cards ---- */
.fg-live-card {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 16px 18px; text-align: center;
}
.fg-live-name { font-size: 0.85rem; font-weight: 600; color: var(--text-mid); margin-bottom: 2px; }
.fg-live-type { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: var(--text-low); margin-bottom: 12px; }
.fg-live-badge { font-size: 1.15rem; font-weight: 700; padding: 10px 0; border-radius: 4px; }
.fg-live-badge.ok { background: var(--safe-dim); color: var(--safe); border: 1px solid rgba(95,174,122,0.3); }
.fg-live-badge.warn { background: var(--critical-dim); color: var(--critical); border: 1px solid rgba(217,105,95,0.35); }
.fg-live-badge.risk { background: rgba(232,169,76,0.12); color: var(--amber); border: 1px solid rgba(232,169,76,0.35); }
.fg-live-source { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: var(--text-low); margin-top: 10px; }

/* ---- Fair-allocation bars ---- */
.fg-bar-row { display: flex; align-items: center; gap: 12px; margin: 9px 0; }
.fg-bar-lbl { width: 96px; font-size: 0.72rem; color: var(--text-mid); font-weight: 600; }
.fg-bar-track {
    flex: 1; height: 16px; background: var(--bg-void); border: 1px solid var(--border);
    border-radius: 3px; overflow: hidden;
}
.fg-bar-fill { height: 100%; min-width: 2px; border-radius: 2px; }
.fg-bar-fill.cyan { background: linear-gradient(90deg, #3D7A96, var(--cyan)); }
.fg-bar-fill.amber { background: linear-gradient(90deg, #7A5A20, var(--amber)); }
.fg-bar-val { width: 118px; text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--text-hi); }
.fg-bars-wrap { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; margin-bottom: 16px; }
.fg-bars-title { font-size: 0.82rem; font-weight: 700; color: var(--text-hi); margin-bottom: 4px; }
.fg-bars-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; color: var(--text-low); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ---- Maintenance timeline ---- */
.fg-tl {
    position: relative; background: var(--bg-void); border: 1px solid var(--border);
    border-radius: 4px; padding: 14px 10px 0 10px; margin: 12px 0 6px 0;
}
.fg-tl-blocks { position: relative; height: 44px; }
.fg-tl-block {
    position: absolute; top: 4px; bottom: 4px; border-radius: 3px;
    border: 1px solid rgba(232,169,76,0.5); background: rgba(232,169,76,0.14);
    display: flex; align-items: center; justify-content: center; overflow: hidden;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--amber);
    white-space: nowrap; text-overflow: clip; padding: 0 4px;
}
.fg-tl-block.active { border-color: var(--amber); background: rgba(232,169,76,0.3); }
.fg-tl-axis { position: relative; height: 16px; margin-top: 2px; }
.fg-tl-tick {
    position: absolute; font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    color: var(--text-low); transform: translateX(-50%); white-space: nowrap;
}

/* ---- History sparkline ---- */
.fg-spark {
    display: flex; align-items: flex-end; gap: 5px; height: 110px; padding: 14px 12px 6px 12px;
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; margin: 10px 0 6px 0;
}
.fg-spark-col { flex: 1; background: linear-gradient(180deg, var(--cyan), #3D7A96); border-radius: 2px 2px 0 0; min-height: 2px; }
.fg-spark-labels { display: flex; gap: 5px; padding: 0 12px; }
.fg-spark-label {
    flex: 1; text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem;
    color: var(--text-low); white-space: nowrap; overflow: hidden;
}

/* ---- Interactive widgets: dark to match the design ---- */
[data-testid="stExpander"] {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
    margin-bottom: 14px;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--text-mid);
}
[data-testid="stExpander"] summary:hover { color: var(--amber); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { border-top: 1px solid var(--border); padding-top: 8px; }

[data-testid="stPills"] button {
    background: var(--bg-panel); border: 1px solid var(--border); color: var(--text-mid);
    border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
}
[data-testid="stPills"] button:hover { border-color: var(--amber); color: var(--amber); }
[data-testid="stPills"] button[aria-pressed="true"] {
    background: var(--amber-dim); border-color: var(--amber); color: var(--amber);
}

[data-testid="stDataFrame"] {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 6px; margin: 8px 0 14px 0;
}
[data-testid="stDataFrame"] * { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.stButton > button, [data-testid="stDownloadButton"] > button {
    background: var(--bg-panel); color: var(--text-hi); border: 1px solid var(--border);
    border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    transition: all 0.2s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    border-color: var(--amber); color: var(--amber); box-shadow: 0 0 20px -4px rgba(242, 200, 121, 0.4);
}

.fg-pills-hint {
    font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; color: var(--text-low);
    text-transform: uppercase; letter-spacing: 0.05em; margin: 14px 0 6px 0;
}

/* ================================================================
   PREMIUM OVERRIDE — obsidian & aurum finish
   ================================================================ */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.mono, .fg-subtitle, .fg-section-label, .fg-badge, .fg-table th,
.fg-live-type, .fg-live-source, .fg-weights-box, .fg-bar-val, .fg-tl-block,
.fg-tl-tick, .fg-spark-label, .fg-pills-hint, .fg-kpi-val, .fg-kpi-sub,
.fg-refresh, .fg-footer { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background-color: var(--bg-void);
    background-image:
        radial-gradient(1100px 560px at 88% -12%, rgba(79, 140, 220, 0.14), transparent 60%),
        radial-gradient(900px 520px at -12% 108%, rgba(232, 180, 90, 0.09), transparent 55%),
        linear-gradient(rgba(122, 152, 207, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(122, 152, 207, 0.045) 1px, transparent 1px);
    background-size: auto, auto, 44px 44px, 44px 44px;
}
.stApp::before {
    content: ''; position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 99999;
    background: linear-gradient(90deg, transparent, var(--amber) 25%, var(--cyan) 50%, var(--amber) 75%, transparent);
    animation: fg-line 8s ease-in-out infinite;
}
@keyframes fg-line { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 6px; border: 2px solid var(--bg-void); }
::selection { background: rgba(242, 200, 121, 0.35); color: var(--text-hi); }

/* ---- Top bar ---- */
.fg-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.05rem; font-weight: 700; letter-spacing: -0.02em; }
.fg-title .fg-lock { color: var(--amber); text-shadow: 0 0 20px rgba(242, 200, 121, 0.55); }
.fg-title-grad {
    background: var(--gold-grad); -webkit-background-clip: text; background-clip: text;
    color: transparent;
}
.fg-subtitle { color: var(--text-mid); letter-spacing: 0.16em; font-size: 0.7rem; margin-top: 8px; }
.fg-status-pill {
    color: var(--safe); background: var(--safe-dim); border: 1px solid rgba(84, 214, 160, 0.4);
    box-shadow: 0 0 24px rgba(84, 214, 160, 0.22), inset 0 0 14px rgba(84, 214, 160, 0.07);
    display: inline-flex; align-items: center; gap: 9px; padding: 8px 16px; border-radius: 4px;
    font-size: 0.68rem; letter-spacing: 0.12em;
}
.fg-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--safe);
    animation: fg-pulse 1.7s ease-in-out infinite;
}
@keyframes fg-pulse {
    0% { box-shadow: 0 0 0 0 rgba(84, 214, 160, 0.55); }
    70% { box-shadow: 0 0 0 9px rgba(84, 214, 160, 0); }
    100% { box-shadow: 0 0 0 0 rgba(84, 214, 160, 0); }
}
.fg-refresh {
    margin-top: 10px; font-size: 0.6rem; color: var(--text-low); letter-spacing: 0.12em; text-align: right;
}

/* ---- KPI row ---- */
.fg-kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 22px 0 10px 0; }
.fg-kpi {
    position: relative; background: var(--panel-grad); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; overflow: hidden; backdrop-filter: blur(10px);
    box-shadow: 0 14px 32px -22px rgba(0, 0, 0, 0.85);
    transition: transform 0.22s ease, border-color 0.22s ease;
}
.fg-kpi::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--kpi-accent, var(--gold-grad)); }
.fg-kpi::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(242, 200, 121, 0.45), transparent);
}
.fg-kpi:hover { transform: translateY(-2px); border-color: var(--border-bright); }
.fg-kpi-val { font-size: 1.9rem; font-weight: 700; color: var(--text-hi); line-height: 1.1; letter-spacing: -0.01em; }
.fg-kpi-lbl { font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-mid); margin-top: 7px; }
.fg-kpi-sub { font-size: 0.62rem; color: var(--text-low); margin-top: 4px; }

/* ---- Section labels ---- */
.fg-section-label { color: var(--amber); font-size: 0.68rem; letter-spacing: 0.2em; }
.fg-section-label::before {
    width: 24px; height: 16px; background: var(--gold-grad); border-radius: 2px;
    box-shadow: 0 0 14px rgba(242, 200, 121, 0.5);
}

/* ---- Flow diagram ---- */
.fg-flow {
    background: var(--panel-grad); border-color: var(--border); backdrop-filter: blur(10px);
    box-shadow: 0 20px 44px -26px rgba(0, 0, 0, 0.85);
}
.fg-flow-node .icon { filter: drop-shadow(0 0 12px rgba(111, 214, 232, 0.45)); }
.fg-flow-node .label { font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; }
.fg-flow-node.local { background: rgba(84, 214, 160, 0.08); border: 1px dashed rgba(84, 214, 160, 0.28); }
.fg-flow-arrow { color: var(--amber); animation: fg-arrow 2.4s ease-in-out infinite; font-size: 1.25rem; }
@keyframes fg-arrow { 0%, 100% { opacity: 0.35; transform: translateX(0); } 50% { opacity: 1; transform: translateX(3px); } }

/* ---- Cards ---- */
.fg-card {
    background: var(--panel-grad); border: 1px solid var(--border); backdrop-filter: blur(12px);
    box-shadow: 0 14px 34px -24px rgba(0, 0, 0, 0.85);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.fg-card:hover {
    transform: translateY(-3px); border-color: var(--border-bright);
    box-shadow: 0 24px 48px -26px rgba(0, 0, 0, 0.9), 0 0 26px -8px rgba(242, 200, 121, 0.2);
}
.fg-card::before {
    width: 3px;
    background: linear-gradient(180deg, transparent,
        var(--card-accent, var(--border-bright)) 40%,
        var(--card-accent, var(--border-bright)) 60%, transparent);
}
.fg-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(242, 200, 121, 0.5), transparent);
}
.fg-card-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.08rem; }
.fg-badge.hw {
    background: rgba(111, 214, 232, 0.10); color: var(--cyan); border: 1px solid rgba(111, 214, 232, 0.35);
    box-shadow: 0 0 14px rgba(111, 214, 232, 0.18);
}
.fg-stat .val { font-size: 1.8rem; background: var(--gold-grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.fg-lock-line {
    background: rgba(84, 214, 160, 0.08); border-color: rgba(84, 214, 160, 0.3);
    box-shadow: inset 0 0 18px rgba(84, 214, 160, 0.05);
}
.fg-weights-box { color: var(--cyan); box-shadow: inset 0 0 22px rgba(111, 214, 232, 0.06); }

/* ---- Aggregation banner ---- */
.fg-agg-banner {
    background: linear-gradient(90deg, rgba(242, 200, 121, 0.12), rgba(242, 200, 121, 0.02));
    border-color: var(--amber-dim); border-left: 3px solid var(--amber);
    box-shadow: 0 0 30px -8px rgba(242, 200, 121, 0.2);
}

/* ---- Tables ---- */
.fg-table th { color: var(--amber); letter-spacing: 0.15em; font-size: 0.62rem; background: rgba(122, 152, 207, 0.06); }
.fg-table td { border-bottom-color: rgba(122, 152, 207, 0.08); }
.fg-table tr:nth-child(even) td { background: rgba(122, 152, 207, 0.03); }
.fg-table tr:hover td { background: rgba(122, 152, 207, 0.08); }

/* ---- Notes ---- */
.fg-note { background: var(--panel-grad); border-color: var(--border); border-left: 3px solid var(--cyan); }

/* ---- Bars ---- */
.fg-bars-wrap { background: var(--panel-grad); border-color: var(--border); backdrop-filter: blur(10px); }
.fg-bar-fill.cyan { background: linear-gradient(90deg, #2E6E8C, var(--cyan)); box-shadow: 0 0 16px rgba(111, 214, 232, 0.45); }
.fg-bar-fill.amber { background: var(--gold-grad); box-shadow: 0 0 16px rgba(242, 200, 121, 0.4); }
.fg-bars-title { font-family: 'Space Grotesk', sans-serif; }

/* ---- Live / maintenance cards ---- */
.fg-live-card { background: var(--panel-grad); border-color: var(--border); backdrop-filter: blur(10px); }

/* ---- Timeline ---- */
.fg-tl { background: rgba(7, 12, 23, 0.6); border-color: var(--border); }
.fg-tl-block {
    border-color: rgba(242, 200, 121, 0.55); background: rgba(242, 200, 121, 0.14);
    color: var(--amber); font-size: 0.62rem; box-shadow: 0 0 14px rgba(242, 200, 121, 0.15);
}
.fg-tl-block.active { background: rgba(242, 200, 121, 0.32); box-shadow: 0 0 20px rgba(242, 200, 121, 0.35); }

/* ---- Sparkline ---- */
.fg-spark { background: var(--panel-grad); border-color: var(--border); }
.fg-spark-col { background: linear-gradient(180deg, var(--cyan), #2E6E8C); box-shadow: 0 0 10px rgba(111, 214, 232, 0.3); }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(16, 24, 42, 0.96), rgba(7, 12, 23, 0.98)); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color: var(--text-hi); font-family: 'Space Grotesk', sans-serif; }

/* ---- Footer ---- */
.fg-footer {
    margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    font-size: 0.6rem; color: var(--text-low); letter-spacing: 0.12em;
}
.fg-footer .gold { color: var(--amber); }
</style>
""", unsafe_allow_html=True)

NODES = ["A", "B", "C"]
NODE_TYPE = {"A": "LIVE HARDWARE", "B": "LIVE HARDWARE", "C": "SIMULATED · SCALE DEMO"}
NODE_IS_HW = {"A": True, "B": True, "C": False}

# ------------------------------------------------------------
# FAIR RESOURCE ALLOCATION — DEFAULTS (edit on the sidebar, no code change)
# ------------------------------------------------------------
# Total shared infrastructure cost for the federation, per month (₹).
# Split proportionally by each node's contributed training samples —
# "you contribute more data, you get a fairer share of the shared cost".
TOTAL_MONTHLY_INFRA_COST_INR = 3000

# Weekly maintenance window per node — when that business's hardware/sensor
# rig gets serviced/recalibrated. Defaults: Sunday, 2-hour slots, 10 AM – 4 PM.
DEFAULT_MAINTENANCE = {
    "A": {"day": "Sunday", "start": datetime.time(10, 0), "end": datetime.time(12, 0)},
    "B": {"day": "Sunday", "start": datetime.time(12, 0), "end": datetime.time(14, 0)},
    "C": {"day": "Sunday", "start": datetime.time(14, 0), "end": datetime.time(16, 0)},
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ------------------------------------------------------------
# FAILURE & COST — per node: detected failure, probable reason,
# and estimated cost = repair + labour + downtime production loss (₹)
# ------------------------------------------------------------
FAILURE_DETAILS = {
    "A": {
        "failure": "None detected",
        "reason": "No anomaly flagged on this node this period.",
        "repair_cost": 0,
        "downtime_hours": 0,
        "loss_per_hour": 0,
    },
    "B": {
        "failure": "None detected",
        "reason": "No anomaly flagged on this node this period.",
        "repair_cost": 0,
        "downtime_hours": 0,
        "loss_per_hour": 0,
    },
    "C": {
        "failure": "Motor bearing failure",
        "reason": "Vibration amplitude exceeded the safe threshold in abnormal windows; bearing raceway wear flagged by the local model.",
        "repair_cost": 4500,     # parts + labour (₹)
        "downtime_hours": 8,     # estimated repair downtime
        "loss_per_hour": 600,    # production loss while down (₹/hr)
    },
}

# ============================================================
# SIDEBAR — live controls for cost share + maintenance windows
# ============================================================
with st.sidebar:
    st.markdown("### Federation Controls")
    st.caption("Fair cost-share is split by data contribution.")
    infra_cost = st.number_input(
        "Monthly infra cost (₹)",
        min_value=0, max_value=100000, value=TOTAL_MONTHLY_INFRA_COST_INR, step=100,
    )

    st.markdown("### Maintenance Windows")
    st.caption("Weekly service slot per node — e.g. Sunday · 2 hrs · 10:00 AM – 12:00 PM")
    maint_schedule = {}
    for node in NODES:
        d = DEFAULT_MAINTENANCE[node]
        st.markdown(f"**Business {node}**")
        day = st.selectbox("Day", WEEKDAYS, index=WEEKDAYS.index(d["day"]), key=f"fg_day_{node}")
        start_t = st.time_input("Start", d["start"], key=f"fg_start_{node}")
        end_t = st.time_input("End", d["end"], key=f"fg_end_{node}")
        maint_schedule[node] = {"day": day, "start": start_t, "end": end_t}

# ============================================================
# HELPERS
# ============================================================
def slot_hours(m):
    s = m["start"].hour * 60 + m["start"].minute
    e = m["end"].hour * 60 + m["end"].minute
    if e <= s:
        e += 24 * 60
    return (e - s) / 60


def slot_label(m):
    return (f"{m['day']} · {m['start'].strftime('%I:%M %p')} – {m['end'].strftime('%I:%M %p')} "
            f"({slot_hours(m):.0f} hr)")


def next_window(m, now=None):
    """Next occurrence of this maintenance window, and whether we are inside it now."""
    now = now or datetime.datetime.now()
    target = WEEKDAYS.index(m["day"])
    days_ahead = (target - now.weekday()) % 7
    next_start = (now + datetime.timedelta(days=days_ahead)).replace(
        hour=m["start"].hour, minute=m["start"].minute, second=0, microsecond=0)
    next_end = next_start.replace(hour=m["end"].hour, minute=m["end"].minute)
    if m["end"] <= m["start"]:
        next_end += datetime.timedelta(days=1)
    if next_start < now:
        next_start += datetime.timedelta(days=7)
        next_end += datetime.timedelta(days=7)
    return next_start, next_end, next_start <= now <= next_end


def countdown_text(delta):
    days, hours, mins = delta.days, delta.seconds // 3600, (delta.seconds % 3600) // 60
    return f"{days}d {hours}h {mins}m"


def failure_cost(f):
    return f["repair_cost"] + f["downtime_hours"] * f["loss_per_hour"]


def load_node_data(node):
    # No caching here on purpose — when you add/update a node's abnormal CSV
    # later, re-running the dashboard must pick up the new data immediately,
    # not serve a stale cached version from before that file existed.
    # Returns (None, None) when a CSV is too short to form even one
    # feature window (needs >= WINDOW_SIZE rows per file).
    normal = pd.read_csv(f"normal_{node}.csv")
    abnormal = pd.read_csv(f"abnormal_{node}.csv")
    X_n = extract_features(normal)
    X_a = extract_features(abnormal)
    if len(X_n) == 0 or len(X_a) == 0:
        return None, None
    X = np.vstack([X_n, X_a])
    y = np.array([0] * len(X_n) + [1] * len(X_a))
    return X, y


def load_local_weights(node):
    try:
        with open(f"weights_node_{node}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def bar_row(label, pct, value_text, css_class="cyan"):
    return f"""
    <div class="fg-bar-row">
        <div class="fg-bar-lbl">{label}</div>
        <div class="fg-bar-track"><div class="fg-bar-fill {css_class}" style="width:{pct:.1f}%;"></div></div>
        <div class="fg-bar-val">{value_text}</div>
    </div>"""


# ============================================================
# TOP BAR
# ============================================================
_refresh_time = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(f"""
<div class="fg-topbar">
    <div>
        <div class="fg-title"><span class="fg-lock">&#128274;</span> <span class="fg-title-grad">FedGuard</span></div>
        <div class="fg-subtitle">FEDERATED PREDICTIVE MAINTENANCE &nbsp;&middot;&nbsp; SHARED MSME EQUIPMENT NETWORK</div>
    </div>
    <div style="text-align:right;">
        <div class="fg-status-pill"><span class="fg-dot"></span> FEDERATION ACTIVE</div>
        <div class="fg-refresh">REFRESHED {_refresh_time}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI ROW — headline numbers at a glance
# ============================================================
_kpi_active = 0
_kpi_samples = 0
for _n in NODES:
    _w = load_local_weights(_n)
    if _w is not None:
        _kpi_active += 1
        _kpi_samples += _w["n_samples"]

_kpi_global = None
try:
    with open("global_model.pkl", "rb") as _f:
        _kpi_global = pickle.load(_f)
except FileNotFoundError:
    _kpi_global = None

_kpi_last_acc = None
try:
    _hist = get_federation_history()
    if _hist is not None and not _hist.empty:
        _kpi_last_acc = float(_hist.iloc[0]["global_accuracy"])
except Exception:
    _kpi_last_acc = None

st.markdown(f"""
<div class="fg-kpi-row">
    <div class="fg-kpi" style="--kpi-accent: var(--gold-grad);">
        <div class="fg-kpi-val">{_kpi_active}/{len(NODES)}</div>
        <div class="fg-kpi-lbl">Active Nodes</div>
        <div class="fg-kpi-sub">local models trained &amp; live</div>
    </div>
    <div class="fg-kpi" style="--kpi-accent: var(--cyan-grad);">
        <div class="fg-kpi-val">{_kpi_samples:,}</div>
        <div class="fg-kpi-lbl">Samples Contributed</div>
        <div class="fg-kpi-sub">fed into the federation</div>
    </div>
    <div class="fg-kpi" style="--kpi-accent: linear-gradient(135deg, #8BE4F4, #54D6A0);">
        <div class="fg-kpi-val">{_kpi_last_acc:.1f}%</div>
        <div class="fg-kpi-lbl">Global Accuracy</div>
        <div class="fg-kpi-sub">last logged federation round</div>
    </div>
    <div class="fg-kpi" style="--kpi-accent: linear-gradient(135deg, #F6D98A, #FF7A6B);">
        <div class="fg-kpi-val">₹{infra_cost:,}</div>
        <div class="fg-kpi-lbl">Infra Cost / Month</div>
        <div class="fg-kpi-sub">fair-split across nodes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIGNATURE ELEMENT — the actual data-flow, not decoration
# ============================================================
st.markdown('<div class="fg-section-label">System Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">What each business keeps local, and what actually crosses the network</div>', unsafe_allow_html=True)

st.markdown("""
<div class="fg-flow">
    <div class="fg-flow-node local">
        <div class="icon">&#128202;</div>
        <div class="label">Raw Vibration Data</div>
        <div class="sub" style="color:var(--critical); font-weight:600;">NEVER TRANSMITTED</div>
    </div>
    <div class="fg-flow-arrow">&#8594;</div>
    <div class="fg-flow-node local">
        <div class="icon">&#9881;&#65039;</div>
        <div class="label">Local Model</div>
        <div class="sub">trains on-device</div>
    </div>
    <div class="fg-flow-arrow">&#8594;</div>
    <div class="fg-flow-node">
        <div class="icon">&#128202;&#65039;</div>
        <div class="label">Weights Only</div>
        <div class="sub" style="color:var(--cyan);">3 numbers, not data</div>
    </div>
    <div class="fg-flow-arrow">&#8594;</div>
    <div class="fg-flow-node">
        <div class="icon">&#128736;&#65039;</div>
        <div class="label">FedAvg Aggregator</div>
        <div class="sub">weighted mean</div>
    </div>
    <div class="fg-flow-arrow">&#8594;</div>
    <div class="fg-flow-node">
        <div class="icon">&#127919;</div>
        <div class="label">Global Model</div>
        <div class="sub">shared back to all</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PER-NODE LOCAL TRAINING PANELS
# ============================================================
st.markdown('<div class="fg-section-label">Node Status — Local Training</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Each business trains independently. Raw data never leaves its own node.</div>', unsafe_allow_html=True)

cols = st.columns(len(NODES))
local_samples = {}

for i, node in enumerate(NODES):
    with cols[i]:
        w = load_local_weights(node)
        badge_class = "hw" if NODE_IS_HW[node] else "sim"
        badge_text = NODE_TYPE[node]

        if w is None:
            st.markdown(f"""
            <div class="fg-card" style="--card-accent: var(--border-bright);">
                <div class="fg-card-head">
                    <div class="fg-card-title">Business {node}</div>
                    <div class="fg-badge {badge_class}">{badge_text}</div>
                </div>
                <div style="color: var(--text-low); font-size: 0.85rem; padding: 20px 0; text-align:center;">
                    No local model trained yet
                </div>
            </div>
            """, unsafe_allow_html=True)
            continue

        from sklearn.linear_model import LogisticRegression
        local_model = LogisticRegression()
        local_model.coef_ = np.array(w["coef"])
        local_model.intercept_ = np.array(w["intercept"])
        local_model.classes_ = np.array([0, 1])

        X, y = load_node_data(node)
        if X is None:
            local_samples[node] = w["n_samples"]
            st.markdown(f"""
            <div class="fg-card" style="--card-accent: var(--amber);">
                <div class="fg-card-head">
                    <div class="fg-card-title">Business {node}</div>
                    <div class="fg-badge {badge_class}">{badge_text}</div>
                </div>
                <div style="color: var(--text-low); font-size: 0.85rem; padding: 14px 0; text-align:center; line-height:1.6;">
                    &#9888;&#65039; CSV too short for feature windows
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--amber); margin-top: 6px;">
                        need &ge; 50 rows in both normal &amp; abnormal files
                    </div>
                    <div style="font-size: 0.74rem; color: var(--text-low); margin-top: 6px;">
                        re-record / re-train via local_model.py
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            continue

        acc = local_model.score(X, y)
        local_samples[node] = w["n_samples"]

        fail = FAILURE_DETAILS.get(node, {"failure": "None detected", "reason": "—",
                                          "repair_cost": 0, "downtime_hours": 0, "loss_per_hour": 0})
        is_abnormal = fail["failure"] != "None detected"
        accent = "var(--critical)" if is_abnormal else ("var(--safe)" if acc >= 0.8 else "var(--amber)")
        status_badge = (
            '<div class="fg-badge" style="color:var(--critical); background:var(--critical-dim);'
            ' border:1px solid rgba(255,122,107,0.5);">&#9888;&#65039; ABNORMAL</div>'
            if is_abnormal else
            '<div class="fg-badge" style="color:var(--safe); background:var(--safe-dim);'
            ' border:1px solid rgba(84,214,160,0.4);">&#10003; OK</div>'
        )
        abnormal_strip = ""
        if is_abnormal:
            abnormal_strip = f"""
        <div style="font-size:0.78rem; color:var(--critical); background:var(--critical-dim);
            border:1px solid rgba(255,122,107,0.35); border-left:3px solid var(--critical);
            border-radius:4px; padding:9px 10px; margin-bottom:10px; line-height:1.55;">
            &#9888;&#65039; <b>FAILURE DETECTED</b> — <b>{fail['failure']}</b><br>
            <span style="color:var(--text-mid);">{fail['reason']}</span><br>
            est. cost <b>&#8377;{failure_cost(fail):,}</b> &nbsp;&middot;&nbsp; {fail['downtime_hours']} hrs downtime
        </div>
        """
        coef_str = ", ".join(f"{v:.2f}" for v in np.array(w["coef"]).flatten())
        intercept_str = f"{np.array(w['intercept']).flatten()[0]:.2f}"

        st.markdown(f"""
        <div class="fg-card" style="--card-accent: {accent};">
            <div class="fg-card-head">
                <div class="fg-card-title">Business {node}</div>
                <div style="display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end;">
                    <div class="fg-badge {badge_class}">{badge_text}</div>
                    {status_badge}
                </div>
            </div>
            <div class="fg-stat-row">
                <div class="fg-stat">
                    <div class="val">{acc*100:.0f}%</div>
                    <div class="lbl">Local Accuracy</div>
                </div>
                <div class="fg-stat">
                    <div class="val">{w['n_samples']}</div>
                    <div class="lbl">Local Samples</div>
                </div>
            </div>
            {abnormal_strip}
        </div>
        """, unsafe_allow_html=True)

        try:
            n_rows_norm = len(pd.read_csv(f"normal_{node}.csv"))
            n_rows_abn = len(pd.read_csv(f"abnormal_{node}.csv"))
        except FileNotFoundError:
            n_rows_norm = n_rows_abn = 0

        with st.expander(f"Details — Business {node}", key=f"fg_exp_{node}"):
            st.markdown(f"""
            <div style="font-size:0.82rem; color:var(--text-mid); line-height:2;">
                <b style="color:var(--text-hi);">Local accuracy</b> &nbsp;·&nbsp; {acc*100:.0f}%<br>
                <b style="color:var(--text-hi);">Samples contributed</b> &nbsp;·&nbsp; {w['n_samples']}<br>
                <b style="color:var(--text-hi);">Data on disk</b> &nbsp;·&nbsp; {n_rows_norm} normal rows / {n_rows_abn} abnormal rows<br>
                <b style="color:var(--text-hi);">Weights sent to aggregator (3 numbers, no raw data)</b><br>
                <span style="color:var(--cyan);">coef: [{coef_str}] &nbsp; intercept: {intercept_str}</span><br>
                <b style="color:var(--text-hi);">Maintenance window</b> &nbsp;·&nbsp; {slot_label(maint_schedule[node])}<br>
                <b style="color:var(--text-hi);">Failure status</b> &nbsp;·&nbsp; {fail['failure']} &nbsp;&middot;&nbsp;
                <span style="color:var(--critical);">est. cost ₹{failure_cost(fail):,}</span>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FEDERATED AGGREGATION
# ============================================================
st.markdown('<div class="fg-section-label">Federated Aggregation — FedAvg</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Central aggregator averages weights only — never accesses raw data from any node.</div>', unsafe_allow_html=True)

per_node_acc_for_log = {}

try:
    with open("global_model.pkl", "rb") as f:
        gm = pickle.load(f)

    nodes_str = ", ".join(f"Business {n}" for n in gm["contributing_nodes"])
    st.markdown(f"""
    <div class="fg-agg-banner">
        &#9989; <b>Global model built</b> from {nodes_str} &nbsp;&middot;&nbsp;
        {gm['total_samples']} samples combined via weight-averaging &nbsp;&middot;&nbsp;
        <b>zero raw data pooled</b>
    </div>
    """, unsafe_allow_html=True)

    global_model = load_global_model("global_model.pkl")

    for node in NODES:
        X, y = load_node_data(node)
        if X is None:
            per_node_acc_for_log[node] = None
            continue
        per_node_acc_for_log[node] = global_model.score(X, y)

    # Log this federation round once per Streamlit session (avoid spamming the DB on every rerun/interaction)
    if "federation_logged" not in st.session_state:
        valid_accs = [a for a in per_node_acc_for_log.values() if a is not None]
        if valid_accs:
            avg_global_acc = float(np.mean(valid_accs)) * 100
            log_federation_round(num_nodes=len(gm["contributing_nodes"]), global_accuracy=avg_global_acc)
        st.session_state["federation_logged"] = True

    acc_pairs = [(local_samples.get(n, 0), per_node_acc_for_log[n])
                 for n in NODES if per_node_acc_for_log.get(n) is not None]
    if acc_pairs:
        total_s = sum(s for s, _ in acc_pairs) or 1
        weighted_acc = sum(s * a for s, a in acc_pairs) / total_s * 100
        simple_acc = float(np.mean([a for _, a in acc_pairs])) * 100
        st.markdown(f"""
        <div class="fg-note" style="margin-bottom:16px;">
            <b>Global accuracy:</b> simple average <span class="mono">{simple_acc:.1f}%</span> &nbsp;&middot;&nbsp;
            sample-weighted (FedAvg weighting) <span class="mono">{weighted_acc:.1f}%</span> — weighted by each
            node's contributed samples
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fg-note">
        <b>Note:</b> the global model can underperform local models on some nodes — this is the
        well-known <b>non-IID data challenge</b> in federated learning (each business's equipment
        behaves a bit differently). This is a real, honest tradeoff of privacy-preserving learning,
        not a bug — in production this motivates adding a personalization layer on top of the global model.
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.markdown("""
    <div class="fg-note" style="border-color: var(--amber-dim);">
        &#9888;&#65039; Global model not built yet — run <span class="mono">federated_server.py</span> first.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FAIR RESOURCE ALLOCATION — fair breakdown, money allocated
# ============================================================
st.markdown('<div class="fg-section-label">Fair Resource Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Cost-share split proportionally by each node\'s data contribution — edit the total on the sidebar</div>', unsafe_allow_html=True)

if local_samples:
    total_samples_all = sum(local_samples.values())

    bars_c1, bars_c2 = st.columns(2)
    with bars_c1:
        bars = ""
        for node in NODES:
            n_s = local_samples.get(node, 0)
            pct = n_s / total_samples_all * 100 if total_samples_all > 0 else 0
            bars += bar_row(f"Business {node}", pct, f"{pct:.1f}% · {n_s} samples")
        st.markdown(f"""
        <div class="fg-bars-wrap">
            <div class="fg-bars-title">Fair breakdown — data contribution</div>
            <div class="fg-bars-sub">samples fed into the federation</div>
            {bars}
        </div>
        """, unsafe_allow_html=True)

    with bars_c2:
        bars = ""
        for node in NODES:
            n_s = local_samples.get(node, 0)
            money = n_s / total_samples_all * infra_cost if total_samples_all > 0 else 0
            pct = money / infra_cost * 100 if infra_cost > 0 else 0
            bars += bar_row(f"Business {node}", pct, f"₹{money:,.0f} / mo", css_class="amber")
        st.markdown(f"""
        <div class="fg-bars-wrap">
            <div class="fg-bars-title">Money allocated — cost share</div>
            <div class="fg-bars-sub">₹{infra_cost:,} / month shared infrastructure</div>
            {bars}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fg-note">
        <b>Basis:</b> the shared infra cost of <b>₹{infra_cost:,}/month</b> is split proportionally to each
        node's contributed training samples — the node that feeds more data into the federation carries a
        proportionally larger (fairer) share of the shared cost. Change the total or the weekly maintenance
        windows on the sidebar — no code edit needed.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="fg-note" style="border-color: var(--amber-dim);">
        &#9888;&#65039; No local models trained yet — allocation table needs at least one node's weights file.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAINTENANCE WINDOWS — when each node is serviced, next slot countdown
# ============================================================
st.markdown('<div class="fg-section-label">Maintenance Windows</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Weekly service slots per node (e.g. Sunday · 2 hrs · 10:00 AM – 12:00 PM) — edit on the sidebar</div>', unsafe_allow_html=True)

now = datetime.datetime.now()
maint_cols = st.columns(len(NODES))

for i, node in enumerate(NODES):
    with maint_cols[i]:
        m = maint_schedule[node]
        nxt_start, nxt_end, in_window = next_window(m, now)
        if in_window:
            badge = "&#9679; MAINTENANCE IN WINDOW NOW"
            badge_color = "var(--amber)"
        else:
            badge = "&#9679; READY"
            badge_color = "var(--safe)"
        st.markdown(f"""
        <div class="fg-live-card">
            <div class="fg-live-name">Business {node}</div>
            <div class="fg-live-type">{slot_label(m)}</div>
            <div class="fg-live-badge" style="font-size:0.95rem; padding:8px 0; color:{badge_color};
                border:1px solid {badge_color}; border-radius:4px; background:rgba(0,0,0,0.25);">{badge}</div>
            <div class="fg-live-source">next: {nxt_start.strftime('%a %d %b · %I:%M %p')} &nbsp;&middot;&nbsp; in {countdown_text(nxt_start - now)}</div>
        </div>
        """, unsafe_allow_html=True)

# Weekly timeline — union of all maintenance slots on one strip
def minutes_of_day(m, rank_offset=0):
    return (WEEKDAYS.index(m["day"]) + rank_offset) * 1440 + m["start"].hour * 60 + m["start"].minute


min_day = min(WEEKDAYS.index(m["day"]) for m in maint_schedule.values())
span_items = []
for node in NODES:
    m = maint_schedule[node]
    s = minutes_of_day(m) - min_day * 1440
    e = s + slot_hours(m) * 60
    span_items.append((node, s, e))

span_start = min(s for _, s, _ in span_items)
span_end = max(e for _, _, e in span_items)
span_total = max(int(span_end - span_start), 1)

tl_blocks = ""
for node, s, e in span_items:
    left = (s - span_start) / span_total * 100
    width = (e - s) / span_total * 100
    active = " active" if next_window(maint_schedule[node], now)[2] else ""
    tl_blocks += f'<div class="fg-tl-block{active}" style="left:{left:.2f}%; width:{width:.2f}%;" title="{slot_label(maint_schedule[node])}">Business {node}</div>'

axis_ticks = ""
for t in range(0, span_total + 1, 30):
    pct = (t - span_start) / span_total * 100 if span_total > 0 else 0
    if pct < 0 or pct > 100:
        continue
    minutes = span_start + t
    day_idx = (minutes // 1440) % 7
    hm = minutes % 1440
    label = f"{WEEKDAYS[day_idx][:3]} {hm//60:02d}:{hm%60:02d}"
    axis_ticks += f'<div class="fg-tl-tick" style="left:{pct:.2f}%;">{label}</div>'

st.markdown(f"""
<div class="fg-tl">
    <div class="fg-tl-blocks">{tl_blocks}</div>
    <div class="fg-tl-axis">{axis_ticks}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fg-pills-hint">Select a node to inspect its maintenance window</div>', unsafe_allow_html=True)
selected_maint = st.pills(
    "Maintenance node",
    options=[f"Business {n}" for n in NODES],
    key="fg_maint_pills",
    label_visibility="collapsed",
)
if selected_maint:
    node = selected_maint.split()[-1]
    m = maint_schedule[node]
    nxt_start, nxt_end, in_window = next_window(m, now)
    status_html = "&#9679; IN WINDOW NOW" if in_window else "&#9679; READY"
    status_color = "var(--amber)" if in_window else "var(--safe)"
    fail = FAILURE_DETAILS.get(node, {})
    st.markdown(f"""
    <div class="fg-card" style="--card-accent: {status_color}; margin-bottom: 16px;">
        <div class="fg-card-head">
            <div class="fg-card-title">{selected_maint} — Maintenance Details</div>
            <div class="fg-badge" style="color:{status_color}; border:1px solid {status_color}; background:rgba(0,0,0,0.25);">{status_html}</div>
        </div>
        <div style="font-size:0.85rem; color:var(--text-mid); line-height:2;">
            <b style="color:var(--text-hi);">Weekly slot</b> &nbsp;·&nbsp; {slot_label(m)}<br>
            <b style="color:var(--text-hi);">Next window</b> &nbsp;·&nbsp; {nxt_start.strftime('%A, %d %b %Y · %I:%M %p')} – {nxt_end.strftime('%I:%M %p')}
            <span style="color:{status_color};">(in {countdown_text(nxt_start - now)})</span><br>
            <b style="color:var(--text-hi);">Duration</b> &nbsp;·&nbsp; {slot_hours(m):.0f} hours<br>
            <b style="color:var(--text-hi);">Failure status</b> &nbsp;·&nbsp; {fail.get('failure', 'None detected')} &nbsp;&middot;&nbsp;
            <span style="color:var(--critical);">est. cost ₹{failure_cost(fail):,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FEDERATION HISTORY — accuracy trend across logged rounds
# ============================================================
st.markdown('<div class="fg-section-label">Federation History</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Global-model accuracy per logged federation round (metrics only, never raw data)</div>', unsafe_allow_html=True)

fed_history = get_federation_history()  # DataFrame, newest first

if fed_history is not None and not fed_history.empty:
    hist_df = fed_history.sort_values("id").reset_index(drop=True)
    hist_df["round"] = range(1, len(hist_df) + 1)
    max_acc = float(hist_df["global_accuracy"].max()) or 1.0

    spark_cols = ""
    labels = ""
    for _, row in hist_df.iterrows():
        acc = float(row["global_accuracy"])
        h = max(acc / max_acc * 100, 2)
        spark_cols += f'<div class="fg-spark-col" style="height:{h:.1f}%;" title="Round {int(row["round"])} · {acc:.1f}%"></div>'
        labels += f'<div class="fg-spark-label">R{int(row["round"])}</div>'

    last = hist_df.iloc[-1]
    st.markdown(f"""
    <div class="fg-note" style="margin-bottom:6px;">
        <b>{len(hist_df)}</b> federation round(s) logged in <span class="mono">federation_history.db</span>.
        Most recent: <span class="mono">{last['timestamp'][:19]}</span> &nbsp;&middot;&nbsp;
        global accuracy <b>{last['global_accuracy']:.1f}%</b>
    </div>
    <div class="fg-spark">{spark_cols}</div>
    <div class="fg-spark-labels">{labels}</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fg-pills-hint">Click a row in the table below to inspect that federation round</div>
    """, unsafe_allow_html=True)

    view_df = hist_df.rename(columns={
        "round": "Round",
        "timestamp": "Timestamp",
        "num_nodes": "Nodes",
        "global_accuracy": "Global Accuracy %",
    })[["Round", "Timestamp", "Nodes", "Global Accuracy %"]]
    sel_state = st.dataframe(
        view_df,
        key="fg_hist_df",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
    )

    selected_rows = sel_state.selection.rows if sel_state else ()
    if selected_rows:
        row = hist_df.iloc[selected_rows[0]]
        r_acc = float(row["global_accuracy"])
        st.markdown(f"""
        <div class="fg-card" style="--card-accent: var(--cyan); margin-bottom: 16px;">
            <div class="fg-card-head">
                <div class="fg-card-title">Round #{int(row['round'])} — details</div>
                <div class="fg-badge hw">SELECTED</div>
            </div>
            <div style="font-size:0.85rem; color:var(--text-mid); line-height:2;">
                <b style="color:var(--text-hi);">Timestamp</b> &nbsp;·&nbsp; {row['timestamp']}<br>
                <b style="color:var(--text-hi);">Nodes contributing</b> &nbsp;·&nbsp; {int(row['num_nodes'])}<br>
                <b style="color:var(--text-hi);">Global accuracy</b> &nbsp;·&nbsp; <span style="color:var(--cyan);">{r_acc:.1f}%</span><br>
                <b style="color:var(--text-hi);">Audit note</b> &nbsp;·&nbsp; metrics only, raw data never logged
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="fg-note">No federation rounds logged yet — run the pipeline at least once.</div>
    """, unsafe_allow_html=True)

# ============================================================
# LOCAL AUDIT DATABASE — full view of federation_history.db
# ============================================================
st.markdown('<div class="fg-section-label">Local Audit Database</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Live read of <span class="mono">federation_history.db</span> — all tables below, metrics only, raw vibration data never stored</div>', unsafe_allow_html=True)

db_local_hist = get_local_history()
db_dataset_loads = get_dataset_loads()
db_fed_hist = fed_history if (fed_history is not None and not fed_history.empty) else None

db_fed_rows = len(db_fed_hist) if db_fed_hist is not None else 0
db_local_rows = len(db_local_hist) if db_local_hist is not None else 0
db_dataset_rows = len(db_dataset_loads) if db_dataset_loads is not None else 0

st.markdown(f"""
<div class="fg-bars-wrap">
    <div class="fg-bars-title">Database snapshot</div>
    <div class="fg-bars-sub">federation_history.db &nbsp;&middot;&nbsp; sqlite &nbsp;&middot;&nbsp; stored locally on this machine</div>
    <div style="display:flex; gap:10px; padding:6px 0 2px 0; flex-wrap:wrap;">
        <span class="fg-badge hw">federation_rounds · {db_fed_rows} rows</span>
        <span class="fg-badge hw">local_history · {db_local_rows} rows</span>
        <span class="fg-badge hw">dataset_loads · {db_dataset_rows} rows</span>
    </div>
</div>
""", unsafe_allow_html=True)

db_tab_fed, db_tab_local, db_tab_loads = st.tabs(
    ["federation_rounds", "local_history", "dataset_loads"]
)

with db_tab_fed:
    if db_fed_rows:
        st.dataframe(
            db_fed_hist.rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "num_nodes": "Nodes",
                "global_accuracy": "Global Accuracy %",
            })[["ID", "Timestamp", "Nodes", "Global Accuracy %"]],
            hide_index=True, width="stretch",
        )
    else:
        st.markdown('<div class="fg-note">No federation rounds logged yet — run the pipeline at least once.</div>', unsafe_allow_html=True)

with db_tab_local:
    if db_local_rows:
        st.dataframe(
            db_local_hist.rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "node": "Node",
                "accuracy": "Accuracy %",
                "n_samples": "Samples",
            })[["ID", "Timestamp", "Node", "Accuracy %", "Samples"]],
            hide_index=True, width="stretch",
        )
    else:
        st.markdown('<div class="fg-note">No local training rounds logged yet — run <span class="mono">local_model.py</span> at least once.</div>', unsafe_allow_html=True)

with db_tab_loads:
    if db_dataset_rows:
        st.dataframe(
            db_dataset_loads.rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "node": "Node",
                "normal_count": "Normal Rows",
                "abnormal_count": "Abnormal Rows",
            })[["ID", "Timestamp", "Node", "Normal Rows", "Abnormal Rows"]],
            hide_index=True, width="stretch",
        )
    else:
        st.markdown('<div class="fg-note">No dataset-load events logged yet — run <span class="mono">local_model.py</span> at least once.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="fg-note">
    <b>Audit trail:</b> every row above is written by <span class="mono">storage.py</span> —
    accuracy metrics and sample counts only. No raw sensor readings, no coefficients, no node identities
    beyond the business label ever enter the database.
</div>
""", unsafe_allow_html=True)

# ============================================================
# COMPLIANCE & WEEKLY REPORT
# ============================================================
st.markdown('<div class="fg-section-label">Compliance &amp; Weekly Report</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Audit trail: aggregate metrics only, never raw vibration data</div>', unsafe_allow_html=True)

def build_weekly_report(history_df):
    lines = []
    lines.append("# FedGuard — Weekly Report")
    lines.append(f"\nGenerated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"\nFederation rounds in this report: {len(history_df)}")
    lines.append("\n## Round History\n")
    lines.append("| Round | Timestamp | Nodes | Global Accuracy |")
    lines.append("|---|---|---|---|")
    for _, row in history_df.sort_values("id").iterrows():
        lines.append(f"| {int(row['round'])} | {row['timestamp'][:19]} | {int(row['num_nodes'])} | {float(row['global_accuracy']):.1f}% |")
    lines.append("\n## Fair Resource Allocation")
    if local_samples:
        total_s = sum(local_samples.values())
        active = len(local_samples)
        equal = infra_cost / active if active else 0
        lines.append(f"\nTotal monthly infra cost: ₹{infra_cost:,}")
        for node in NODES:
            n_s = local_samples.get(node, 0)
            pct = n_s / total_s * 100 if total_s > 0 else 0
            money = n_s / total_s * infra_cost if total_s > 0 else 0
            vs = equal - money
            lines.append(f"- Business {node}: {pct:.1f}% share, ₹{money:,.0f}/mo "
                         f"({'+' if vs >= 0 else ''}₹{abs(vs):,.0f} vs equal split), "
                         f"maintenance {slot_label(maint_schedule[node])}")
    lines.append("\n## Privacy Note")
    lines.append("\nThis report contains only aggregate accuracy metrics and sample counts. "
                 "No raw vibration data from any business is stored in this log or included in this report. "
                 "No email or cloud (ThingSpeak) services are used — the report is generated locally.")
    return "\n".join(lines)

report_text = build_weekly_report(hist_df) if (fed_history is not None and not fed_history.empty) else "No rounds logged yet."

st.download_button(
    label="Download Weekly Report (.md)",
    data=report_text,
    file_name=f"fedguard_report_{datetime.date.today().isoformat()}.md",
    mime="text/markdown",
)

st.markdown("""
<div class="fg-footer">
    <span><span class="gold">FEDGUARD</span> · FEDERATED PREDICTIVE MAINTENANCE</span>
    <span>METRICS LOGGED LOCALLY · ZERO RAW DATA TRANSMITTED</span>
</div>
""", unsafe_allow_html=True)
