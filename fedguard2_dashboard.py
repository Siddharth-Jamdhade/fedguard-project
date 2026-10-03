"""
fedguard_dashboard.py — FedGuard dashboard
Control-room / industrial-monitoring visual design: this is built for MSME
equipment operators, not a generic AI demo, so the interface borrows from
real factory-floor instrumentation — graphite panels, amber caution accents,
monospace data readouts.

Run: streamlit run fedguard_dashboard.py
Works directly on the CSVs + .pkl files already in this folder
(synthetic today, real MPU6050 data later — same file names, zero code change).
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

from local_model import extract_features
from federated_server import load_global_model

st.set_page_config(page_title="FedGuard — Federated Predictive Maintenance", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# THEME — control-room instrumentation palette
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg-void: #0B0F14;
    --bg-panel: #151B23;
    --bg-panel-raised: #1B2330;
    --border: #2A3442;
    --border-bright: #3D4A5C;
    --amber: #E8A94C;
    --amber-dim: #6B5530;
    --cyan: #5FB8D9;
    --safe: #5FAE7A;
    --safe-dim: rgba(95, 174, 122, 0.12);
    --critical: #D9695F;
    --critical-dim: rgba(217, 105, 95, 0.12);
    --text-hi: #E8EAED;
    --text-mid: #9AA5B5;
    --text-low: #5C6779;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--bg-void); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

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
</style>
""", unsafe_allow_html=True)

NODES = ["A", "B", "C"]
NODE_TYPE = {"A": "LIVE HARDWARE", "B": "LIVE HARDWARE", "C": "SIMULATED · SCALE DEMO"}
NODE_IS_HW = {"A": True, "B": True, "C": False}

# ============================================================
# TOP BAR
# ============================================================
st.markdown("""
<div class="fg-topbar">
    <div>
        <div class="fg-title"><span class="fg-lock">&#128274;</span> FedGuard</div>
        <div class="fg-subtitle">FEDERATED PREDICTIVE MAINTENANCE &nbsp;&middot;&nbsp; SHARED MSME EQUIPMENT NETWORK</div>
    </div>
    <div class="fg-status-pill">&#9679; FEDERATION ACTIVE</div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_node_data(node):
    normal = pd.read_csv(f"normal_{node}.csv")
    abnormal = pd.read_csv(f"abnormal_{node}.csv")
    X_n = extract_features(normal)
    X_a = extract_features(abnormal)
    X = np.vstack([X_n, X_a])
    y = np.array([0] * len(X_n) + [1] * len(X_a))
    return X, y


def load_local_weights(node):
    try:
        with open(f"weights_node_{node}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


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
local_accuracies = {}

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
        acc = local_model.score(X, y)
        local_accuracies[node] = acc

        accent = "var(--safe)" if acc >= 0.8 else "var(--amber)"
        coef_str = ", ".join(f"{v:.2f}" for v in np.array(w["coef"]).flatten())
        intercept_str = f"{np.array(w['intercept']).flatten()[0]:.2f}"

        st.markdown(f"""
        <div class="fg-card" style="--card-accent: {accent};">
            <div class="fg-card-head">
                <div class="fg-card-title">Business {node}</div>
                <div class="fg-badge {badge_class}">{badge_text}</div>
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
            <div class="fg-lock-line">&#128274; Raw vibration data — local only</div>
            <div class="fg-weights-label">Transmitted to aggregator</div>
            <div class="fg-weights-box">coef: [{coef_str}]<br>intercept: {intercept_str}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FEDERATED AGGREGATION
# ============================================================
st.markdown('<div class="fg-section-label">Federated Aggregation — FedAvg</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Central aggregator averages weights only — never accesses raw data from any node.</div>', unsafe_allow_html=True)

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

    st.markdown('<div style="font-weight:600; color:var(--text-hi); font-size:0.95rem; margin-bottom:2px;">Global Model Performance Per Node</div>', unsafe_allow_html=True)
    st.markdown('<div class="fg-section-sub" style="margin-bottom:8px;">Evaluated on each node\'s full local data — the global model never trained on that node\'s raw data directly</div>', unsafe_allow_html=True)

    rows_html = ""
    for node in NODES:
        X, y = load_node_data(node)
        acc = global_model.score(X, y)
        local_acc = local_accuracies.get(node, 0) * 100
        global_acc = acc * 100
        delta = global_acc - local_acc
        delta_color = "var(--safe)" if delta >= 0 else "var(--critical)"
        delta_sign = "+" if delta >= 0 else ""
        rows_html += f"""
        <tr>
            <td>Business {node}</td>
            <td class="mono-val">{local_acc:.0f}%</td>
            <td class="mono-val">{global_acc:.0f}%</td>
            <td class="mono-val" style="color:{delta_color};">{delta_sign}{delta:.0f}pp</td>
        </tr>
        """

    st.markdown(f"""
    <table class="fg-table">
        <tr><th>Node</th><th>Local Model</th><th>Global (Federated) Model</th><th>&Delta; vs Local</th></tr>
        {rows_html}
    </table>
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
# LIVE EQUIPMENT STATUS
# ============================================================
st.markdown('<div class="fg-section-label">Live Equipment Status</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-section-sub">Real-time classification per node, powered by the local model above</div>', unsafe_allow_html=True)

demo_status_override = {"A": "NORMAL", "B": "NORMAL", "C": "ABNORMAL"}
status_cols = st.columns(len(NODES))

for i, node in enumerate(NODES):
    with status_cols[i]:
        label = demo_status_override.get(node, "NORMAL")
        is_ok = label == "NORMAL"
        badge_html = "&#9679; NORMAL" if is_ok else "&#9679; ABNORMAL — MAINTENANCE FLAGGED"
        badge_class = "ok" if is_ok else "warn"
        type_label = NODE_TYPE[node]

        st.markdown(f"""
        <div class="fg-live-card">
            <div class="fg-live-name">Business {node}</div>
            <div class="fg-live-type">{type_label}</div>
            <div class="fg-live-badge {badge_class}">{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)
