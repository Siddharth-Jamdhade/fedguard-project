"""
FlowGuard - FINAL LIVE ESP32 HC-SR04 BLOCKAGE DETECTOR

Physical setup
--------------
Sensor empty/reference height : 13.53 cm
Known clean pipe area         : 2.83 cm^2  (1.90 cm diameter pipe)
ESP32 Serial                  : 115200 baud

ESP32 sends:
    t_ms,distance_cm,water_level_cm

LIVE METHOD
-----------
1. Start this program with the pipe/channel CLEAN.
2. Keep the inflow condition the same.
3. The program collects 20 clean water-level readings.
4. It calculates the clean baseline water height.
5. For every later live reading:
       blockage % =
       (1 - sqrt(clean_height / current_height)) * 100

This relative formulation is used for the LIVE blockage percentage
because the same inflow condition is assumed before/after blockage.
It avoids the incorrect hard-coded 0.100 L/s assumption that was
producing impossible effective areas (17-25 cm^2 for a 2.83 cm^2 pipe).

It also reports a relative effective area:
       A_effective = A_clean * sqrt(clean_height / current_height)

For the clean baseline:
       A_effective ≈ 2.83 cm^2
       blockage ≈ 0%

When the obstruction raises water level:
       current_height > clean_height
       A_effective decreases
       blockage % increases

Run:
    python blockage_detector1.py --port COM7
"""

import argparse
import time
from collections import deque

import serial
from serial.tools import list_ports


# ============================================================
# REAL PHYSICAL SETTINGS
# ============================================================

SENSOR_MOUNT_HEIGHT_CM = 13.53
# Cross-sectional area of a 1.90 cm DIAMETER pipe:
# A = pi * (d/2)^2 = 3.14159 * 0.95^2 = 2.83 cm^2
CLEAN_AREA_CM2 = 2.83

BAUD_RATE = 115200
SERIAL_TIMEOUT_S = 2.0

# Number of clean readings used to establish the baseline.
BASELINE_READINGS = 20

# Moving-average smoothing for HC-SR04 noise.
SMOOTHING_WINDOW = 5

# ML-style sustained confirmation.
CONFIRMATION_READINGS = 4
CONFIRMATION_THRESHOLD_PCT = 10.0

# Ignore almost-zero head because the relative formula becomes
# meaningless for a dry channel.
MIN_HEAD_CM = 0.10


# ============================================================
# LIVE STATE
# ============================================================

level_history = deque(maxlen=SMOOTHING_WINDOW)
blockage_history = deque(maxlen=120)

clean_height_cm = None


# ============================================================
# SERIAL PORT
# ============================================================

def find_esp32_port():
    ports = list(list_ports.comports())

    if not ports:
        return None

    keywords = (
        "ESP32",
        "USB",
        "CP210",
        "CH340",
        "CH341",
        "Silicon Labs",
        "USB-SERIAL",
    )

    for port in ports:
        text = (
            f"{port.description} "
            f"{port.manufacturer or ''}"
        )

        if any(
            keyword.lower() in text.lower()
            for keyword in keywords
        ):
            return port.device

    if len(ports) == 1:
        return ports[0].device

    return None


def open_serial(port="auto"):
    if port.lower() == "auto":
        port = find_esp32_port()

    if not port:
        available = [
            p.device
            for p in list_ports.comports()
        ]

        raise RuntimeError(
            "ESP32 COM port not detected. "
            f"Available ports: {available}. "
            "Run with --port COM7."
        )

    print(
        f"Connecting to ESP32 on {port} "
        f"at {BAUD_RATE} baud..."
    )

    ser = serial.Serial(
        port=port,
        baudrate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT_S,
    )

    # Opening the port may reset the ESP32.
    time.sleep(2)
    ser.reset_input_buffer()

    print(f"CONNECTED: {port}")

    return ser


# ============================================================
# ESP32 CSV PARSER
# ============================================================

def parse_sensor_line(line):
    """
    Expected:
        t_ms,distance_cm,water_level_cm

    Python recalculates water level as:

        water_level = 13.53 - distance
    """

    line = line.strip()

    if not line:
        return None

    if line.lower().startswith("t_ms"):
        return None

    parts = line.split(",")

    if len(parts) != 3:
        return None

    try:
        t_ms = int(float(parts[0]))
        distance_cm = float(parts[1])
        _firmware_water_level = float(parts[2])
    except ValueError:
        return None

    if distance_cm < 0:
        return None

    water_level_cm = max(
        0.0,
        SENSOR_MOUNT_HEIGHT_CM - distance_cm,
    )

    return (
        t_ms,
        distance_cm,
        water_level_cm,
    )


# ============================================================
# SENSOR SMOOTHING
# ============================================================

def smooth_level(level_cm):
    level_history.append(level_cm)

    return sum(level_history) / len(
        level_history
    )


# ============================================================
# CLEAN BASELINE
# ============================================================

def collect_clean_baseline(ser):
    """
    Collect 20 clean readings with NO obstruction.

    Use the median for a robust baseline.
    """

    print()
    print("=" * 56)
    print(" CLEAN-CHANNEL BASELINE")
    print("=" * 56)
    print()
    print("REMOVE THE OBSTRUCTION.")
    print("Keep the same inflow condition you will use during")
    print("the blockage demonstration.")
    print()
    print(
        f"Collecting {BASELINE_READINGS} valid readings..."
    )
    print()

    samples = []

    while len(samples) < BASELINE_READINGS:

        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        parsed = parse_sensor_line(line)

        if parsed is None:
            continue

        _, distance_cm, water_level_cm = parsed

        samples.append(water_level_cm)

        print(
            f"Clean "
            f"{len(samples):02d}/{BASELINE_READINGS} | "
            f"distance={distance_cm:6.2f} cm | "
            f"water={water_level_cm:6.2f} cm"
        )

    sorted_samples = sorted(samples)

    if len(sorted_samples) % 2 == 0:
        mid = len(sorted_samples) // 2

        baseline = (
            sorted_samples[mid - 1]
            + sorted_samples[mid]
        ) / 2.0

    else:
        baseline = sorted_samples[
            len(sorted_samples) // 2
        ]

    if baseline < MIN_HEAD_CM:
        raise RuntimeError(
            "Clean water level is too low for blockage "
            "calculation."
        )

    return baseline


# ============================================================
# REAL BLOCKAGE CALCULATION
# ============================================================

def calculate_live_blockage(
    current_height_cm,
    baseline_height_cm,
):
    """
    Relative hydraulic blockage estimate.

    With the same inflow condition:

        A_current / A_clean
            = sqrt(h_clean / h_current)

    Therefore:

        blockage =
            (1 - sqrt(h_clean / h_current)) * 100

    This gives:
        clean condition -> ~0%
        higher water     -> positive blockage
    """

    if current_height_cm < MIN_HEAD_CM:
        return None, None

    if baseline_height_cm < MIN_HEAD_CM:
        return None, None

    area_ratio = (
        baseline_height_cm
        / current_height_cm
    ) ** 0.5

    area_ratio = max(
        0.0,
        min(1.0, area_ratio),
    )

    effective_area = (
        CLEAN_AREA_CM2
        * area_ratio
    )

    blockage = (
        1.0 - area_ratio
    ) * 100.0

    blockage = max(
        0.0,
        min(100.0, blockage),
    )

    return (
        effective_area,
        blockage,
    )


# ============================================================
# SUSTAINED CONFIRMATION
# ============================================================

def confirmation_status(blockage_pct):
    if blockage_pct is None:
        return False

    blockage_history.append(
        blockage_pct
    )

    if len(blockage_history) < CONFIRMATION_READINGS:
        return False

    recent = list(
        blockage_history
    )[-CONFIRMATION_READINGS:]

    return all(
        value >= CONFIRMATION_THRESHOLD_PCT
        for value in recent
    )


# ============================================================
# LIVE PROCESSING
# ============================================================

def process_live_stream(ser):
    global clean_height_cm

    print()
    print("=" * 56)
    print(" FLOWGUARD LIVE ESP32 BLOCKAGE DETECTOR")
    print("=" * 56)
    print(
        f"Sensor empty height : "
        f"{SENSOR_MOUNT_HEIGHT_CM:.2f} cm"
    )
    print(
        f"Known clean area    : "
        f"{CLEAN_AREA_CM2:.2f} cm^2"
    )
    print()
    print(
        "No artificial Q, Cd or 12 cm^2 area is used for "
        "the live blockage percentage."
    )

    clean_height_cm = collect_clean_baseline(
        ser
    )

    # Reset smoothing state after baseline.
    level_history.clear()
    blockage_history.clear()

    print()
    print("=" * 56)
    print(" BASELINE READY")
    print("=" * 56)
    print(
        f"Clean water height : "
        f"{clean_height_cm:.3f} cm"
    )
    print(
        f"Clean area         : "
        f"{CLEAN_AREA_CM2:.3f} cm^2"
    )
    print(
        "Clean blockage     : approximately 0.00%"
    )
    print()
    print(
        "NOW insert the REAL obstruction."
    )
    print(
        "The water level should increase and the blockage"
    )
    print(
        "percentage should increase accordingly."
    )
    print()
    print(
        "Press Ctrl+C to stop."
    )
    print()

    while True:

        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        parsed = parse_sensor_line(
            line
        )

        if parsed is None:
            continue

        (
            t_ms,
            distance_cm,
            water_level_cm,
        ) = parsed

        live_level = smooth_level(
            water_level_cm
        )

        effective_area, blockage = (
            calculate_live_blockage(
                live_level,
                clean_height_cm,
            )
        )

        confirmed = confirmation_status(
            blockage
        )

        if effective_area is None:
            area_text = "N/A"
        else:
            area_text = (
                f"{effective_area:7.2f} cm^2"
            )

        if blockage is None:
            blockage_text = "N/A"
        else:
            blockage_text = (
                f"{blockage:6.2f}%"
            )

        if confirmed:
            status = "BLOCKAGE CONFIRMED"
        elif (
            blockage is not None
            and blockage >= 5.0
        ):
            status = "BLOCKAGE DETECTED"
        else:
            status = "normal/monitoring"

        print(
            f"[LIVE] "
            f"t={t_ms:7d} ms | "
            f"distance={distance_cm:6.2f} cm | "
            f"water={live_level:6.2f} cm | "
            f"effective_area={area_text} | "
            f"blockage={blockage_text} | "
            f"{status}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "FlowGuard live ESP32 HC-SR04 "
            "blockage detector"
        )
    )

    parser.add_argument(
        "--port",
        default="auto",
        help="ESP32 COM port, e.g. COM7",
    )

    args = parser.parse_args()

    try:
        ser = open_serial(
            args.port
        )

    except Exception as exc:
        print()
        print("ERROR:")
        print(exc)
        return

    try:
        process_live_stream(
            ser
        )

    except KeyboardInterrupt:
        print()
        print(
            "Stopping FlowGuard live detector..."
        )

    finally:
        ser.close()

        print(
            "ESP32 serial port closed."
        )


if __name__ == "__main__":
    main()