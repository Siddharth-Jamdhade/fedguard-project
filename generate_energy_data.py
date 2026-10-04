"""
generate_energy_data.py — Create realistic demo CSVs for PS-13 FedGuard.

Generates energy_A.csv, energy_B.csv, energy_C.csv
(same column format as the Wokwi ESP32 sketch output).

Each CSV = 200 readings @ 15-min intervals ≈ 50 hours (≈ 1 week of operation).

Business profiles:
  A — Factory / Manufacturing   : high load 600-800 W, mechanical fault near end
  B — Office / Commercial       : moderate load 250-350 W, electrical anomaly mid-run
  C — Workshop / Light Industry : variable 400-550 W, mostly normal with mild wear

Usage:
    python generate_energy_data.py
"""

import numpy as np
import pandas as pd

N = 200          # readings per node
INTERVAL = 15    # minutes per reading (15-min smart-meter interval)
np.random.seed(42)


def make_node(name: str,
              base_power: float,
              noise_w: float,
              base_vib: float,
              noise_vib: float,
              anomaly_start: int,
              anomaly_type: str,   # "mechanical" | "electrical" | "wear" | "none"
              seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_ms = np.arange(N) * INTERVAL * 60 * 1000  # ms

    # ── Normal baseline ───────────────────────────────────────────────────────
    power = rng.normal(base_power, noise_w, N).clip(50, 2500)
    vib   = rng.normal(base_vib,   noise_vib, N).clip(0.05, 10)

    # Add gentle diurnal cycle: power slightly lower at start/end
    diurnal = np.sin(np.linspace(0, np.pi, N)) * noise_w * 0.5
    power += diurnal

    # ── Inject anomaly ────────────────────────────────────────────────────────
    n_anom = N - anomaly_start

    if anomaly_type == "mechanical":
        # Gradual power AND vibration rise: bearing degradation
        ramp = np.linspace(0, 1, n_anom)
        power[anomaly_start:] += ramp * base_power * 0.65 + rng.normal(0, noise_w, n_anom)
        vib[anomaly_start:]   += ramp * base_vib   * 2.2  + rng.normal(0, noise_vib, n_anom)

    elif anomaly_type == "electrical":
        # Power spikes only, vibration stays normal: PF degradation
        ramp = np.linspace(0, 1, n_anom)
        power[anomaly_start:] += ramp * base_power * 0.55 + rng.normal(0, noise_w * 1.2, n_anom)
        # vibration unchanged

    elif anomaly_type == "wear":
        # Vibration creep only, power barely affected: early bearing wear
        ramp = np.linspace(0, 1, n_anom)
        vib[anomaly_start:] += ramp * base_vib * 1.3 + rng.normal(0, noise_vib, n_anom)
        power[anomaly_start:] += rng.normal(0, noise_w * 0.3, n_anom)

    power = power.clip(50, 2500)
    vib   = vib.clip(0.05, 10)

    # ── Build ax, ay, az consistent with vibration_rms ───────────────────────
    # Distribute vibration across axes (gravity dominates az)
    az = 9.80 + rng.normal(0, 0.02, N)            # gravity
    xy_component = np.sqrt(np.maximum(0, vib**2 - az**2) / 2)
    ax = xy_component + rng.normal(0, noise_vib * 0.1, N)
    ay = xy_component + rng.normal(0, noise_vib * 0.1, N)
    vib_rms = np.sqrt(ax**2 + ay**2 + az**2)

    df = pd.DataFrame({
        "timestamp_ms":  t_ms.astype(int),
        "power_w":       np.round(power, 1),
        "ax":            np.round(ax, 4),
        "ay":            np.round(ay, 4),
        "az":            np.round(az, 4),
        "vibration_rms": np.round(vib_rms, 4),
    })

    fname = f"energy_{name}.csv"
    df.to_csv(fname, index=False)
    anom_label = anomaly_type if anomaly_start < N else "none"
    print(f"✓ Node {name}: {fname}  ({N} readings @ {INTERVAL}min | anomaly={anom_label} from reading {anomaly_start})")
    return df


if __name__ == "__main__":
    print("--- Generating PS-13 demo energy CSVs ---")
    make_node("A", base_power=700, noise_w=45,  base_vib=9.82, noise_vib=0.08,
              anomaly_start=140, anomaly_type="mechanical", seed=101)

    make_node("B", base_power=290, noise_w=18,  base_vib=9.81, noise_vib=0.04,
              anomaly_start=110, anomaly_type="electrical",  seed=202)

    make_node("C", base_power=470, noise_w=35,  base_vib=9.83, noise_vib=0.10,
              anomaly_start=165, anomaly_type="wear",         seed=303)

    print()
    print("Next steps:")
    print("  python energy_local_model.py --node A --csv energy_A.csv")
    print("  python energy_local_model.py --node B --csv energy_B.csv")
    print("  python energy_local_model.py --node C --csv energy_C.csv")
    print("  python energy_federated_server.py --nodes weights_energy_A.pkl weights_energy_B.pkl weights_energy_C.pkl")
    print("  streamlit run energy_dashboard.py")
