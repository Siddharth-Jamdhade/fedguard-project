"""
live_monitor.py — reads live CSV from the ESP32 over serial and computes
blockage % correctly, in real time.

KEY FIX vs. your earlier script: Q (flow rate) is now MEASURED from the
actual rate of change of water level, not assumed/hardcoded. This is why
your previous version kept showing "blockage" even with the obstacle
removed — it was using a stale/wrong Q value that didn't match the real
(often near-zero, static) flow at that moment.

Setup required before running:
  1. Measure your tank/lake's cross-sectional area (width x length, ruler,
     in cm^2) -> set A_TANK_CM2 below.
  2. Your pipe diameter is 1.90 cm -> A_CLEAN is calculated automatically
     below from that, using the SAME formula every time (fixes any
     mismatch between calibration and live detection).
  3. Your sensor mount height (measured earlier) -> SENSOR_MOUNT_HEIGHT_CM.
  4. You must have already calibrated Cd once (see blockage_detector.py's
     calibrate_cd()) -> paste that value into CD_CALIBRATED below.

Usage:
    pip install pyserial
    python live_monitor.py --port COM7
"""

import argparse
import time
import math
import serial

from blockage_detector import calculate_area, blockage_percent

# ------------------------------------------------------------------
# FILL THESE IN WITH YOUR REAL MEASURED VALUES
# ------------------------------------------------------------------
SENSOR_MOUNT_HEIGHT_CM = 13.53  # your measured value, empty-tank sensor reading

PIPE_DIAMETER_CM = 1.90
A_CLEAN_CM2 = math.pi * (PIPE_DIAMETER_CM / 2) ** 2  # = 2.835 cm^2, computed ONCE, used everywhere

A_TANK_CM2 = None  # <-- SET THIS: measure your tank/lake's width x length with a ruler (cm^2)
# Example: if your tank is roughly 20cm x 15cm at the waterline, A_TANK_CM2 = 300.0

CD_CALIBRATED = None  # <-- SET THIS: paste in the Cd value you got from calibrate_cd()
# Example: CD_CALIBRATED = 0.62

BLOCKAGE_ALERT_THRESHOLD_PCT = 15.0  # only flag as "BLOCKAGE DETECTED" above this %
MIN_FLOW_RATE_TO_ANALYZE = 0.05  # cm/s — below this, treat water as "static", skip analysis
# ------------------------------------------------------------------


def parse_line(line):
    """Expects CSV: t_ms,distance_cm,water_level_cm (matches ESP32 firmware output)."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        t_ms = float(parts[0])
        distance_cm = float(parts[1])
        water_level_cm = float(parts[2])
        return t_ms, distance_cm, water_level_cm
    except ValueError:
        return None  # header line or garbage, skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. COM7 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    if A_TANK_CM2 is None:
        raise SystemExit("Set A_TANK_CM2 at the top of this script first — measure your tank's cross-sectional area.")
    if CD_CALIBRATED is None:
        raise SystemExit("Set CD_CALIBRATED at the top of this script first — run calibrate_cd() and paste the result here.")

    print(f"A_CLEAN (from {PIPE_DIAMETER_CM}cm diameter pipe): {A_CLEAN_CM2:.3f} cm^2")
    print(f"Using Cd = {CD_CALIBRATED}, A_TANK = {A_TANK_CM2} cm^2")
    print("Connecting...")

    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(2)  # let ESP32 reset after serial connect

    prev_t_sec = None
    prev_h = None

    while True:
        raw_line = ser.readline().decode("utf-8", errors="ignore")
        parsed = parse_line(raw_line)
        if parsed is None:
            continue

        t_ms, distance_cm, water_level_cm = parsed
        t_sec = t_ms / 1000.0
        h = water_level_cm

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

        area = calculate_area(q_actual, CD_CALIBRATED, h)
        pct = blockage_percent(area, A_CLEAN_CM2)

        status = "CLEAR"
        if pct is not None:
            if pct > BLOCKAGE_ALERT_THRESHOLD_PCT:
                status = "BLOCKAGE DETECTED"
            elif pct > 5:
                status = "MINOR / WITHIN NOISE"

        area_str = f"{area:.2f}" if area is not None else "N/A"
        pct_str = f"{pct:.2f}%" if pct is not None else "N/A"

        print(f"[LIVE] t={t_ms:.0f}ms | distance={distance_cm:.2f}cm | water={h:.2f}cm | "
              f"Q_measured={q_actual:.2f}cm3/s | effective_area={area_str}cm^2 | "
              f"blockage={pct_str} | {status}")

        prev_t_sec, prev_h = t_sec, h


if __name__ == "__main__":
    main()