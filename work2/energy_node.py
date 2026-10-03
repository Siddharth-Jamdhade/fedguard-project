"""
energy_node.py — runs locally at EACH business, same trust boundary as
local_model.py. Reads power-sensor data, never sends raw readings
anywhere; only ever contributes to the same weights_node_*.pkl file
local_model.py already produces (power becomes one more feature
alongside rx, ry, rz).

Two input modes:
  --serial PORT   Read live "adc_raw,power_w" lines from the Wokwi/ESP32
                   serial port (see the firmware + wiring notes).
  --csv PATH       Replay a recorded energy_data.csv (useful with no
                   hardware attached — this is the default demo mode).

Usage:
    python energy_node.py --node Business_A --csv Business_A/energy_data.csv
    python energy_node.py --node Business_A --serial COM5
"""

import argparse
import csv
import sys
import time

from carbon_calculator import summarize_period
from anomaly_explainer import diagnose


def read_csv_rows(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row


def read_serial_rows(port, baud=115200):
    import serial  # pip install pyserial
    ser = serial.Serial(port, baud, timeout=2)
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            yield {"adc_raw": parts[0], "power_w": parts[1]}


def adc_to_watts(adc_raw, max_adc=4095, max_watts=2500):
    """Map the Wokwi potentiometer's 0-4095 ADC reading to a simulated 0-2500W load."""
    return round((float(adc_raw) / max_adc) * max_watts, 1)


def run(node_name, rows, interval_minutes=15, baseline_power_w=250, baseline_vibration_rms=0.3):
    readings = []
    log_path = f"{node_name}_energy_log.csv"
    report_path = f"{node_name}_anomaly_report.csv"

    with open(log_path, "w", newline="") as logf, open(report_path, "w", newline="") as repf:
        log_writer = csv.writer(logf)
        log_writer.writerow(["timestamp", "power_w", "cost_inr_so_far", "co2_kg_so_far"])
        rep_writer = csv.writer(repf)
        rep_writer.writerow(["timestamp", "severity", "likely_cause", "suggested_action",
                              "estimated_monthly_savings_inr", "estimated_monthly_co2_reduction_kg"])

        for row in rows:
            if "power_w" in row and row["power_w"]:
                power_w = float(row["power_w"])
            elif "adc_raw" in row:
                power_w = adc_to_watts(row["adc_raw"])
            else:
                continue

            readings.append(power_w)
            summary = summarize_period(readings, interval_minutes)
            log_writer.writerow([row.get("timestamp", time.time()), power_w,
                                  summary["cost_inr"], summary["co2_kg"]])

            vibration_rms = float(row.get("vibration_rms", baseline_vibration_rms))
            diag = diagnose(power_w, baseline_power_w, vibration_rms, baseline_vibration_rms)
            if diag["severity"] != "none":
                rep_writer.writerow([row.get("timestamp", time.time()), diag["severity"],
                                      diag["likely_cause"], diag["suggested_action"],
                                      diag["estimated_monthly_savings_inr"],
                                      diag["estimated_monthly_co2_reduction_kg"]])

    print(f"[{node_name}] wrote {log_path} and {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--csv", help="Path to a recorded energy_data.csv")
    parser.add_argument("--serial", help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    args = parser.parse_args()

    if args.csv:
        rows = read_csv_rows(args.csv)
    elif args.serial:
        rows = read_serial_rows(args.serial)
    else:
        print("Pass --csv PATH or --serial PORT", file=sys.stderr)
        sys.exit(1)

    run(args.node, rows)
