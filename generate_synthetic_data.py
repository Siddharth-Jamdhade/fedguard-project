"""
generate_synthetic_data.py — creates fake but realistic MPU6050-style CSVs
so you can build/test local_model.py and federated_server.py before any
hardware is wired. Same CSV format the real ESP32 firmware will produce
(t_ms, ax, ay, az), so swapping in real data later requires zero code changes
elsewhere.

Usage:
    python generate_synthetic_data.py
    (creates normal_A.csv, abnormal_A.csv, normal_B.csv, abnormal_B.csv,
     normal_C.csv, abnormal_C.csv)
"""

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 50
DURATION_SEC = 60
N_SAMPLES = SAMPLE_RATE_HZ * DURATION_SEC

np.random.seed(42)


def generate_node_data(node_name, base_noise, imbalance_freq, imbalance_amp, seed_offset):
    """
    base_noise: baseline vibration noise level for this node (simulates each
                node's equipment having a slightly different normal signature)
    imbalance_freq: frequency of the induced wobble in 'abnormal' state (Hz)
    imbalance_amp: amplitude of the abnormal wobble
    """
    rng = np.random.default_rng(42 + seed_offset)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE_HZ

    # ---- NORMAL: gravity on z (~9.8) + small random noise on all axes ----
    ax_n = rng.normal(0, base_noise, N_SAMPLES)
    ay_n = rng.normal(0, base_noise, N_SAMPLES)
    az_n = 9.8 + rng.normal(0, base_noise, N_SAMPLES)

    normal_df = pd.DataFrame({
        "t_ms": (t * 1000).astype(int),
        "ax": ax_n, "ay": ay_n, "az": az_n,
    })

    # ---- ABNORMAL: same baseline + added periodic wobble (simulated imbalance) ----
    wobble = imbalance_amp * np.sin(2 * np.pi * imbalance_freq * t)
    ax_a = rng.normal(0, base_noise, N_SAMPLES) + wobble
    ay_a = rng.normal(0, base_noise, N_SAMPLES) + wobble * 0.6
    az_a = 9.8 + rng.normal(0, base_noise * 1.5, N_SAMPLES)  # more vertical noise too

    abnormal_df = pd.DataFrame({
        "t_ms": (t * 1000).astype(int),
        "ax": ax_a, "ay": ay_a, "az": az_a,
    })

    normal_df.to_csv(f"normal_{node_name}.csv", index=False)
    abnormal_df.to_csv(f"abnormal_{node_name}.csv", index=False)
    print(f"Node {node_name}: wrote normal_{node_name}.csv and abnormal_{node_name}.csv")


if __name__ == "__main__":
    # Each node gets slightly different characteristics — mimics real MSMEs'
    # equipment having genuinely different usage/vibration signatures,
    # which is exactly why federating (rather than pooling raw data) is useful.
    generate_node_data("A", base_noise=0.15, imbalance_freq=6, imbalance_amp=1.2, seed_offset=1)
    generate_node_data("B", base_noise=0.20, imbalance_freq=8, imbalance_amp=1.5, seed_offset=2)
    generate_node_data("C", base_noise=0.12, imbalance_freq=5, imbalance_amp=1.0, seed_offset=3)

    print("\nDone. Now run, for each node:")
    print("  python local_model.py --node A --normal normal_A.csv --abnormal abnormal_A.csv")
    print("  python local_model.py --node B --normal normal_B.csv --abnormal abnormal_B.csv")
    print("  python local_model.py --node C --normal normal_C.csv --abnormal abnormal_C.csv")
    print("\nThen:")
    print("  python federated_server.py --nodes weights_node_A.pkl weights_node_B.pkl weights_node_C.pkl")
