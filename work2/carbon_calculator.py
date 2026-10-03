"""
carbon_calculator.py — cost + carbon footprint estimation (PS-13 core requirement)

Plugs into the existing FedGuard pipeline alongside local_model.py.
No raw data leaves a node because of this file — it only runs on
numbers a node already has locally (its own power readings).

Grid CO2 emission factor source:
  CEA "CO2 Baseline Database for the Indian Power Sector" —
  weighted average grid emission factor = 0.727 kgCO2e/kWh.
  (CEA updates this periodically; swap INDIA_GRID_EMISSION_FACTOR_KGCO2_PER_KWH
  for the latest published value before a real deployment.)

Tariff slabs below are a placeholder flat industrial/commercial rate —
replace with your local DISCOM's actual slab tariff (e.g. MSEDCL for
Nagpur/Maharashtra) for a production system.
"""

INDIA_GRID_EMISSION_FACTOR_KGCO2_PER_KWH = 0.727

# Placeholder flat tariff — swap for your DISCOM's real slab rates.
DEFAULT_TARIFF_INR_PER_KWH = 9.0


def power_w_to_kwh(power_w, duration_hours):
    """Convert an instantaneous power reading (W) over a duration (hours) to kWh."""
    return (power_w * duration_hours) / 1000.0


def estimate_cost(kwh, tariff_inr_per_kwh=DEFAULT_TARIFF_INR_PER_KWH):
    return round(kwh * tariff_inr_per_kwh, 2)


def estimate_co2_kg(kwh, emission_factor=INDIA_GRID_EMISSION_FACTOR_KGCO2_PER_KWH):
    return round(kwh * emission_factor, 3)


def summarize_period(readings_w, interval_minutes, tariff_inr_per_kwh=DEFAULT_TARIFF_INR_PER_KWH):
    """
    readings_w: list of power readings in Watts, sampled every `interval_minutes`.
    Returns total kWh, cost (INR), and CO2 (kg) for that period — this is what
    feeds the dashboard's per-business, per-period comparison view.
    """
    hours_per_reading = interval_minutes / 60.0
    total_kwh = sum(power_w_to_kwh(w, hours_per_reading) for w in readings_w)
    return {
        "total_kwh": round(total_kwh, 3),
        "cost_inr": estimate_cost(total_kwh, tariff_inr_per_kwh),
        "co2_kg": estimate_co2_kg(total_kwh),
    }


def compare_periods(period_a_summary, period_b_summary, label_a="Period A", label_b="Period B"):
    """Used for the 'compares usage across multiple spaces or time periods' requirement."""
    delta_kwh = round(period_b_summary["total_kwh"] - period_a_summary["total_kwh"], 3)
    pct_change = (
        round((delta_kwh / period_a_summary["total_kwh"]) * 100, 1)
        if period_a_summary["total_kwh"] else 0.0
    )
    return {
        "comparison": f"{label_b} vs {label_a}",
        "delta_kwh": delta_kwh,
        "pct_change": pct_change,
        "delta_cost_inr": round(period_b_summary["cost_inr"] - period_a_summary["cost_inr"], 2),
        "delta_co2_kg": round(period_b_summary["co2_kg"] - period_a_summary["co2_kg"], 3),
    }
