"""
live_monitor.py — reads live CSV from the ESP32 over serial and computes
blockage % in real time, using ALL THREE layers from blockage_detector.py:

  1. PHYSICS       — calculate_area() + blockage_percent()
                      (orifice/Manning's-derived live area calc)
  2. ML CONFIRMATION — extract_window_features() + BlockageAnomalyDetector
                      (Isolation Forest flags sustained pattern shifts,
                       not single noisy spikes)
  3. TREND FORECAST  — forecast_days_to_critical()
                      (projects "days until critical blockage" from the
                       accumulated blockage-% log)

v3 CHANGE vs earlier version: this file previously only used Layer 1.
It now imports and runs all three layers exactly as defined in
blockage_detector.py — nothing in that file was modified.

------------------------------------------------------------------
YOUR MEASURED PHYSICAL SETUP (plugged in below):
  Pipe diameter : 1.93 cm  -> A_CLEAN computed automatically from this
  Tank area     : 308 cm^2
  Calibrated Cd : 0.542
------------------------------------------------------------------

HOW THE ML LAYER GETS ITS BASELINE:
Isolation Forest needs to see "known-clean" windows before it can flag
anomalies. This script collects the first BASELINE_WINDOWS windows of
blockage-% readings (assuming your channel starts CLEAR/unblocked) and
fits the detector on those. Only after that does it start flagging
confirmed anomalies. Make sure the channel is actually clear when you
start the script for this baseline to mean anything.

HOW THE FORECAST LAYER WORKS LIVE:
Every confirmed blockage-% reading gets appended to blockage_log.csv
(timestamp in days, blockage %). forecast_days_to_critical() re-fits a
trend line over that accumulated log on every new reading and reports
projected days-to-critical once there's enough history (3+ points).
Note: over a single short live session this trend will be very noisy —
it becomes meaningful once the log spans hours/days of real deployment.

Setup required before running:
  1. Your sensor mount height (measured earlier) -> SENSOR_MOUNT_HEIGHT_CM.
  2. You must have already calibrated Cd once (see blockage_detector.py's
     calibrate_cd()) -> already plugged in below as CD_CALIBRATED.

Usage:
    pip install pyserial numpy scikit-learn
    python live_monitor.py --port COM7
"""

import argparse
import csv
import os
import time
import math
import serial

from blockage_detector import (
    calculate_area,
    blockage_percent,
    extract_window_features,
    BlockageAnomalyDetector,
    forecast_days_to_critical,
)

# ------------------------------------------------------------------
# FILL THESE IN WITH YOUR REAL MEASURED VALUES
# ------------------------------------------------------------------
SENSOR_MOUNT_HEIGHT_CM = 13.53  # your measured value, empty-tank sensor reading

PIPE_DIAMETER_CM = 1.93
A_CLEAN_CM2 = math.pi * (PIPE_DIAMETER_CM / 2) ** 2  # computed ONCE, used everywhere

A_TANK_CM2 = 308  # measured tank/lake cross-sectional area, cm^2

CD_CALIBRATED = 0.542  # from calibrate_cd()

BLOCKAGE_ALERT_THRESHOLD_PCT = 15.0  # raw physics threshold for "BLOCKAGE DETECTED"
MIN_FLOW_RATE_TO_ANALYZE = 0.05  # cm/s — below this, treat water as "static", skip analysis

# --- ML confirmation layer settings ---
WINDOW_SIZE = 5           # readings per rolling window fed to the ML layer
BASELINE_WINDOWS = 20     # how many clean windows to collect before fitting the detector
ANOMALY_CONTAMINATION = 0.05  # expected fraction of anomalies in "normal" data

# --- Trend forecast layer settings ---
CRITICAL_THRESHOLD_PCT = 50.0   # blockage % considered "critical"
LOG_FILE = "blockage_log.csv"   # accumulated (day, blockage_pct) history for forecasting
# ------------------------------------------------------------------


def parse_line(line):
    """Expects CSV: t_ms,distance_cm,water_level_cm (matches ESP32 firmware output).
    Debug lines starting with "#" and the CSV header line are safely rejected here
    since they won't split into exactly 3 numeric fields."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        t_ms = float(parts[0])
        distance_cm = float(parts[1])
        water_level_cm = float(parts[2])
    except ValueError:
        return None  # header line or garbage, skip

    # Defensive check: reject invalid/error readings (firmware sends -1 on
    # no-echo). Without this, a stray -1 could look like a sudden huge water
    # level drop and corrupt the flow-rate calculation.
    if distance_cm < 0 or water_level_cm < 0:
        return None

    return t_ms, distance_cm, water_level_cm


def append_to_log(day_ts, pct):
    """Appends one (day, blockage_pct) row to the running CSV log used by
    the forecast layer. Creates the file with a header if it doesn't exist yet."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["day_ts", "blockage_pct"])
        writer.writerow([f"{day_ts:.6f}", f"{pct:.4f}"])


def load_log():
    """Reads the full accumulated log back for forecasting."""
    days, pcts = [], []
    if not os.path.isfile(LOG_FILE):
        return days, pcts
    with open(LOG_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) != 2:
                continue
            try:
                days.append(float(row[0]))
                pcts.append(float(row[1]))
            except ValueError:
                continue
    return days, pcts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. COM7 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--quiet", action="store_true", help="suppress [RAW] debug lines")
    args = parser.parse_args()

    if A_TANK_CM2 is None:
        raise SystemExit("Set A_TANK_CM2 at the top of this script first — measure your tank's cross-sectional area.")
    if CD_CALIBRATED is None:
        raise SystemExit("Set CD_CALIBRATED at the top of this script first — run calibrate_cd() and paste the result here.")

    print(f"A_CLEAN (from {PIPE_DIAMETER_CM}cm diameter pipe): {A_CLEAN_CM2:.3f} cm^2")
    print(f"Using Cd = {CD_CALIBRATED}, A_TANK = {A_TANK_CM2} cm^2")
    print(f"Opening {args.port} @ {args.baud} baud...")

    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(2)  # let ESP32 reset after serial connect
    print("Connected. Waiting for data...\n")
    print(f"[ML] Collecting {BASELINE_WINDOWS} baseline windows (channel should be CLEAR right now)...\n")

    prev_t_sec = None
    prev_h = None

    # Rolling window + ML confirmation state
    current_window = []          # blockage % readings building up to WINDOW_SIZE
    baseline_features = []       # feature vectors collected during baseline phase
    detector = BlockageAnomalyDetector(contamination=ANOMALY_CONTAMINATION)
    detector_ready = False

    session_start_sec = None

    while True:
        raw_bytes = ser.readline()
        if not raw_bytes:
            # timeout (2s) with nothing received at all — port is open but
            # ESP32 isn't sending ANYTHING. Usually wrong baud rate or the
            # board isn't running/flashed correctly.
            print("[WARN] No data received in 2s — check baud rate matches Serial.begin() on the ESP32, "
                  "and that no other program (Serial Monitor etc.) has the port open.")
            continue

        raw_line = raw_bytes.decode("utf-8", errors="ignore")
        if not args.quiet and raw_line.strip():
            print(f"[RAW] {raw_line.strip()}")

        parsed = parse_line(raw_line)
        if parsed is None:
            continue

        t_ms, distance_cm, water_level_cm = parsed
        t_sec = t_ms / 1000.0
        h = water_level_cm

        if session_start_sec is None:
            session_start_sec = t_sec

        if prev_t_sec is None:
            # first reading, nothing to compare against yet
            prev_t_sec, prev_h = t_sec, h
            print(f"[LIVE] t={t_ms:.0f}ms | distance={distance_cm:.2f}cm | water={h:.2f}cm | (warming up, need 2 readings to compute flow)")
            continue

        dt = t_sec - prev_t_sec
        dh = h - prev_h

        if dt <= 0:
            continue  # duplicate/out-of-order timestamp, skip

        rate_of_change = dh / dt  # cm/s, negative if draining, positive if filling

        # Q_actual: how much water is actually flowing OUT through the pipe
        # right now, measured from how fast the tank level is dropping.
        # (If level is rising, water is flowing IN faster than it drains —
        # can't isolate outflow cleanly in that case, so we skip analysis.)
        if rate_of_change >= -MIN_FLOW_RATE_TO_ANALYZE:
            # water is static or rising — no reliable outflow to analyze
            print(f"[LIVE] t={t_ms:.0f}ms | distance={distance_cm:.2f}cm | water={h:.2f}cm | "
                  f"NO ACTIVE OUTFLOW (level static/rising) — skipping blockage calc")
            prev_t_sec, prev_h = t_sec, h
            continue

        q_actual = A_TANK_CM2 * abs(rate_of_change)  # cm^3/s, measured from real drainage speed

        # --------------------------------------------------------
        # LAYER 1 — PHYSICS
        # --------------------------------------------------------
        area = calculate_area(q_actual, CD_CALIBRATED, h)
        pct = blockage_percent(area, A_CLEAN_CM2)

        area_str = f"{area:.2f}" if area is not None else "N/A"
        pct_str = f"{pct:.2f}%" if pct is not None else "N/A"

        raw_status = "CLEAR"
        if pct is not None:
            if pct > BLOCKAGE_ALERT_THRESHOLD_PCT:
                raw_status = "BLOCKAGE DETECTED (raw threshold)"
            elif pct > 5:
                raw_status = "MINOR / WITHIN NOISE"

        ml_status = "n/a (collecting baseline)"

        if pct is not None:
            # --------------------------------------------------------
            # LAYER 2 — ML CONFIRMATION (rolling window)
            # --------------------------------------------------------
            current_window.append(pct)
            if len(current_window) >= WINDOW_SIZE:
                features = extract_window_features(current_window[-WINDOW_SIZE:])

                if not detector_ready:
                    baseline_features.append(features)
                    remaining = BASELINE_WINDOWS - len(baseline_features)
                    ml_status = f"n/a (baseline {len(baseline_features)}/{BASELINE_WINDOWS})"
                    if remaining <= 0:
                        detector.fit(baseline_features)
                        detector_ready = True
                        print(f"\n[ML] Baseline collected — anomaly detector is now ACTIVE.\n")
                else:
                    is_anomaly = detector.is_confirmed_anomaly(features)
                    ml_status = "CONFIRMED ANOMALY (sustained pattern)" if is_anomaly else "normal pattern"

            # --------------------------------------------------------
            # LAYER 3 — TREND FORECAST (accumulated log)
            # --------------------------------------------------------
            day_ts = (t_sec - session_start_sec) / 86400.0  # elapsed time, in days
            append_to_log(day_ts, pct)
            days_log, pcts_log = load_log()
            forecast = forecast_days_to_critical(pcts_log, days_log, critical_threshold_pct=CRITICAL_THRESHOLD_PCT)
            forecast_str = f"{forecast} days" if forecast is not None else "flat/insufficient data"

        else:
            forecast_str = "n/a"

        print(f"[LIVE] t={t_ms:.0f}ms | distance={distance_cm:.2f}cm | water={h:.2f}cm | "
              f"Q_measured={q_actual:.2f}cm3/s | effective_area={area_str}cm^2 | "
              f"blockage={pct_str} | physics={raw_status} | ml={ml_status} | "
              f"forecast_to_{CRITICAL_THRESHOLD_PCT:.0f}pct={forecast_str}")

        prev_t_sec, prev_h = t_sec, h


if __name__ == "__main__":
    main()
