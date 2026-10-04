"""
energy_carbon.py — PS-13 Cost & CO₂ Calculator

India-specific constants:
  Tariff     : ₹ 8.00 / kWh  (MSME industrial, flat rate)
  Emission   : 0.82 kg CO₂ / kWh  (CEA Grid Emission Factor, 2023)
  Working    : 200 hours / month   (≈ 25 days × 8 h)
"""

TARIFF_INR_PER_KWH = 8.0
EMISSION_FACTOR_KG_CO2_PER_KWH = 0.82
DEFAULT_MONTHLY_HOURS = 200


def watts_hours_to_kwh(watts: float, hours: float) -> float:
    return watts * hours / 1000.0


def cost_inr(kwh: float, tariff: float = TARIFF_INR_PER_KWH) -> float:
    return round(kwh * tariff, 4)


def co2_kg(kwh: float, factor: float = EMISSION_FACTOR_KG_CO2_PER_KWH) -> float:
    return round(kwh * factor, 4)


def summarize_period(power_w_readings: list, interval_minutes: float = 15.0) -> dict:
    """
    Given a list of power (W) readings (each covering interval_minutes),
    return energy, cost, CO₂ and peak stats.
    """
    if not power_w_readings:
        return {"energy_kwh": 0, "cost_inr": 0, "co2_kg": 0,
                "avg_power_w": 0, "peak_power_w": 0, "n_readings": 0}

    hours_per_reading = interval_minutes / 60.0
    energy_per_reading = [w * hours_per_reading / 1000.0 for w in power_w_readings]
    total_kwh = sum(energy_per_reading)

    return {
        "energy_kwh":    round(total_kwh, 4),
        "cost_inr":      cost_inr(total_kwh),
        "co2_kg":        co2_kg(total_kwh),
        "avg_power_w":   round(sum(power_w_readings) / len(power_w_readings), 1),
        "peak_power_w":  round(max(power_w_readings), 1),
        "min_power_w":   round(min(power_w_readings), 1),
        "n_readings":    len(power_w_readings),
    }


def compare_periods(readings_a: list, readings_b: list,
                    interval_minutes: float = 15.0) -> dict:
    """Compare two time periods (e.g. first half vs second half of a day)."""
    sa = summarize_period(readings_a, interval_minutes)
    sb = summarize_period(readings_b, interval_minutes)

    delta_kwh  = sb["energy_kwh"] - sa["energy_kwh"]
    delta_cost = sb["cost_inr"]   - sa["cost_inr"]
    delta_co2  = sb["co2_kg"]     - sa["co2_kg"]
    pct = (delta_kwh / sa["energy_kwh"] * 100) if sa["energy_kwh"] > 0 else 0.0

    return {
        "period_a":          sa,
        "period_b":          sb,
        "delta_energy_kwh":  round(delta_kwh, 4),
        "delta_cost_inr":    round(delta_cost, 4),
        "delta_co2_kg":      round(delta_co2, 4),
        "pct_change":        round(pct, 1),
    }


def project_monthly(avg_power_w: float,
                    monthly_hours: float = DEFAULT_MONTHLY_HOURS) -> dict:
    """Project from an average power reading to monthly cost + CO₂."""
    kwh = watts_hours_to_kwh(avg_power_w, monthly_hours)
    return {
        "projected_monthly_kwh":     round(kwh, 2),
        "projected_monthly_cost_inr": cost_inr(kwh),
        "projected_monthly_co2_kg":   co2_kg(kwh),
    }
