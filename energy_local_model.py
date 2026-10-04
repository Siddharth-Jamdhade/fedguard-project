"""
energy_local_model.py — PS-13 FedGuard local training (run once per node).

Reads a CSV produced by the Wokwi ESP32 (or generate_energy_data.py).
Extracts rolling-window features, labels windows as normal/anomalous,
trains a LogisticRegression, then saves ONLY the model weights (not raw data).

Usage:
    python energy_local_model.py --node A --csv energy_A.csv
    python energy_local_model.py --node B --csv energy_B.csv
    python energy_local_model.py --node C --csv energy_C.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from energy_storage import log_energy_round

# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW_SIZE    = 5      # readings per feature window
BASELINE_FRAC  = 0.30   # fraction of data used to define "normal" baseline
POWER_RATIO    = 1.35   # power > 1.35× baseline → anomalous label
VIB_RATIO      = 1.40   # vibration > 1.40× baseline → anomalous label
TEST_SIZE      = 0.25


# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Skip comment lines (lines starting with '#')
    df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].copy()
    df = df.reset_index(drop=True)

    # Normalise column names (Wokwi may output slightly different headers)
    df.columns = [c.strip().lower() for c in df.columns]

    # Compute vibration_rms from ax/ay/az if not already present
    if "vibration_rms" not in df.columns:
        if all(c in df.columns for c in ["ax", "ay", "az"]):
            df["vibration_rms"] = np.sqrt(
                df["ax"].astype(float)**2 +
                df["ay"].astype(float)**2 +
                df["az"].astype(float)**2
            )
        else:
            df["vibration_rms"] = 9.81   # fallback: gravity-only

    df["power_w"]        = df["power_w"].astype(float)
    df["vibration_rms"]  = df["vibration_rms"].astype(float)
    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Sliding-window feature extraction: 8 features per window."""
    feats = []
    pw  = df["power_w"].values
    vib = df["vibration_rms"].values
    n   = len(df)

    for i in range(0, n - WINDOW_SIZE + 1, WINDOW_SIZE):
        wp = pw[i : i + WINDOW_SIZE]
        wv = vib[i : i + WINDOW_SIZE]
        x  = np.arange(len(wp), dtype=float)
        slope = np.polyfit(x, wp, 1)[0] if len(wp) > 1 else 0.0

        feats.append([
            wp.mean(),                          # mean power
            wp.std() if len(wp) > 1 else 0.0,  # power variability
            wp.max(),                           # peak power
            wp.max() - wp.min(),                # power range
            slope,                              # power trend
            wv.mean(),                          # mean vibration
            wv.std() if len(wv) > 1 else 0.0,  # vibration variability
            wv.max(),                           # peak vibration
        ])
    return np.array(feats, dtype=float)


def make_labels(df: pd.DataFrame):
    """Auto-label windows using first BASELINE_FRAC of data as normal reference."""
    bn = max(1, int(len(df) * BASELINE_FRAC))
    bp = df["power_w"].iloc[:bn].mean()
    bv = df["vibration_rms"].iloc[:bn].mean()

    labels = []
    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        win = df.iloc[i : i + WINDOW_SIZE]
        ap  = win["power_w"].mean()
        av  = win["vibration_rms"].mean()
        anomaly = (ap > bp * POWER_RATIO) or (av > bv * VIB_RATIO)
        labels.append(1 if anomaly else 0)

    return np.array(labels), float(bp), float(bv)


def main():
    parser = argparse.ArgumentParser(
        description="PS-13 FedGuard: train local energy anomaly model")
    parser.add_argument("--node", required=True,
                        help="Node identifier, e.g. A  (used in weights filename)")
    parser.add_argument("--csv",  required=True,
                        help="Path to energy CSV (from Wokwi or generate_energy_data.py)")
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"  Node {args.node} — Local Training")
    print(f"  Input CSV : {args.csv}")
    print(f"{'─'*60}")

    df = load_csv(args.csv)
    print(f"  Loaded {len(df)} readings")

    X = extract_features(df)
    y, baseline_power, baseline_vib = make_labels(df)

    print(f"  Windows   : {len(X)}  |  anomalous windows: {y.sum()}")
    print(f"  Baseline  : power={baseline_power:.1f} W,  vib={baseline_vib:.4f}")

    # Ensure at least 2 classes before split
    if len(np.unique(y)) < 2:
        print("  [WARN] Only one class in labels — forcing minority class on last 2 windows.")
        y[-1] = 1
        y[-2] = 1

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    ts = TEST_SIZE if len(X) >= 8 else 0.2
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ts, random_state=42)
        tr_idx, te_idx = next(sss.split(X_sc, y))
        X_tr, X_te = X_sc[tr_idx], X_sc[te_idx]
        y_tr, y_te = y[tr_idx],    y[te_idx]
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=ts, random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)

    print(f"  Local model accuracy : {acc:.3f}  (test size={len(X_te)})")

    # ── Persist to storage (audit trail) ─────────────────────────────────────
    log_energy_round(
        node=args.node,
        accuracy=acc,
        n_samples=len(X_tr),
        baseline_power_w=baseline_power,
        baseline_vib=baseline_vib,
    )

    # ── Save ONLY weights — raw CSV stays on this node ────────────────────────
    out = f"weights_energy_{args.node}.pkl"
    weights = {
        "node":                    args.node,
        "coef":                    model.coef_.tolist(),
        "intercept":               model.intercept_.tolist(),
        "n_samples":               len(X_tr),
        "baseline_power_w":        baseline_power,
        "baseline_vibration_rms":  baseline_vib,
        "scaler_mean":             scaler.mean_.tolist(),
        "scaler_scale":            scaler.scale_.tolist(),
        "accuracy":                acc,
    }
    with open(out, "wb") as f:
        pickle.dump(weights, f)

    print(f"\n  ✓ Saved {out}")
    print(f"  ✎ Only {out} leaves this node — raw CSV stays local.\n")


if __name__ == "__main__":
    main()
