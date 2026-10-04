"""
energy_anomaly.py — PS-13 Anomaly Explainer

Uses Motor Current Signature Analysis (MCSA) logic:
  • Power spike  + Vibration spike  → mechanical fault (bearing/rotor)
  • Power spike  + Vibration normal → electrical inefficiency (power factor)
  • Power normal + Vibration spike  → early-stage mechanical wear
  • Both normal                     → no anomaly

Returns severity, likely_cause, suggested_action, and estimated savings.
"""

from energy_carbon import TARIFF_INR_PER_KWH, EMISSION_FACTOR_KG_CO2_PER_KWH

# Thresholds (multiples of baseline)
POWER_THRESH_HIGH   = 1.50   # >50% above baseline → high severity
POWER_THRESH_MEDIUM = 1.25   # >25% above baseline → medium
POWER_THRESH_LOW    = 1.10   # >10% above baseline → low

VIB_THRESH_HIGH   = 1.80
VIB_THRESH_MEDIUM = 1.40
VIB_THRESH_LOW    = 1.15

DEFAULT_MONTHLY_HOURS = 200  # working hours/month


def diagnose(power_w: float, baseline_power_w: float,
             vibration_rms: float, baseline_vibration_rms: float,
             tariff: float = TARIFF_INR_PER_KWH,
             monthly_hours: float = DEFAULT_MONTHLY_HOURS) -> dict:
    """
    Run MCSA diagnosis on one window of readings.

    Returns dict with:
      severity, likely_cause, suggested_action,
      estimated_monthly_savings_inr, estimated_monthly_co2_reduction_kg
    """
    if baseline_power_w <= 0:
        baseline_power_w = max(power_w, 1)
    if baseline_vibration_rms <= 0:
        baseline_vibration_rms = max(vibration_rms, 0.01)

    pr = power_w / baseline_power_w
    vr = vibration_rms / baseline_vibration_rms

    excess_w = max(0.0, power_w - baseline_power_w)
    excess_kwh_monthly = excess_w * monthly_hours / 1000.0
    savings_inr = round(excess_kwh_monthly * tariff, 0)
    co2_reduction = round(excess_kwh_monthly * EMISSION_FACTOR_KG_CO2_PER_KWH, 2)

    # ── No anomaly ──────────────────────────────────────────────────────────────
    if pr < POWER_THRESH_LOW and vr < VIB_THRESH_LOW:
        return _result("none",
            "Normal operation — power and vibration within expected range.",
            "No action required. Continue monitoring.",
            0, 0)

    # ── BOTH elevated: mechanical fault (MCSA pattern) ─────────────────────────
    if pr >= POWER_THRESH_HIGH and vr >= VIB_THRESH_HIGH:
        return _result("high",
            "Mechanical fault detected — motor drawing excess current alongside high vibration. "
            "Classic MCSA signature of bearing damage or rotor imbalance.",
            "Schedule immediate maintenance. Inspect bearings, check rotor balance, "
            "lubricate or replace worn parts. Consider a shutdown window in the next 48 hours.",
            savings_inr, co2_reduction)

    if pr >= POWER_THRESH_MEDIUM and vr >= VIB_THRESH_MEDIUM:
        return _result("high",
            "Combined power + vibration rise — developing mechanical fault. "
            "Motor efficiency is degrading; likely early bearing wear or shaft misalignment.",
            "Book a preventive maintenance slot within 1 week. Check coupling alignment "
            "and bearing condition. Clean motor vents to rule out thermal overload.",
            savings_inr, co2_reduction)

    # ── Power high, vibration normal: electrical issue ─────────────────────────
    if pr >= POWER_THRESH_HIGH and vr < VIB_THRESH_MEDIUM:
        return _result("medium",
            "Electrical inefficiency — high power draw without abnormal vibration. "
            "Likely power factor degradation, phase imbalance, or a reactive load issue. "
            "Mechanical components appear intact.",
            "Audit switchgear for phase imbalance. Inspect and test power factor correction "
            "capacitor banks. Consider installing an automatic PFC unit (typical payback < 1 year).",
            savings_inr, co2_reduction)

    if pr >= POWER_THRESH_MEDIUM and vr < VIB_THRESH_LOW:
        return _result("medium",
            "Moderate power overconsumption with normal vibration — possible load scheduling "
            "issue, over-spec motor running lightly loaded, or voltage sag compensation.",
            "Check if a smaller motor could serve the same load. Audit off-peak vs peak usage "
            "scheduling. Verify supply voltage is within ±5% of rated.",
            savings_inr, co2_reduction)

    # ── Vibration high, power normal: early mechanical wear ────────────────────
    if vr >= VIB_THRESH_HIGH and pr < POWER_THRESH_MEDIUM:
        return _result("medium",
            "High vibration without significant power increase — early-stage mechanical "
            "degradation. Worn bearings or misalignment before it starts drawing excess current.",
            "Schedule preventive maintenance within 2 weeks. Lubricate bearings and check "
            "coupling. Acting now avoids a higher-severity fault and unplanned downtime.",
            round(savings_inr * 0.4, 0), round(co2_reduction * 0.4, 2))

    if vr >= VIB_THRESH_MEDIUM and pr < POWER_THRESH_MEDIUM:
        return _result("low",
            "Elevated vibration within tolerance — possible minor imbalance or resonance. "
            "Power consumption is still normal.",
            "Log and monitor for recurrence. If vibration remains elevated over the next "
            "2–3 days, inspect the motor mount and coupling for looseness.",
            round(savings_inr * 0.2, 0), round(co2_reduction * 0.2, 2))

    # ── Low-level mixed anomaly ────────────────────────────────────────────────
    return _result("low",
        "Minor deviation from baseline — could be load variation, startup surge, or "
        "measurement noise. Not yet a confirmed fault.",
        "Continue monitoring. Flag for review if this pattern repeats consistently "
        "across more than 3 consecutive windows.",
        round(savings_inr * 0.1, 0), round(co2_reduction * 0.1, 2))


def _result(severity, cause, action, savings_inr, co2_reduction):
    return {
        "severity": severity,
        "likely_cause": cause,
        "suggested_action": action,
        "estimated_monthly_savings_inr": savings_inr,
        "estimated_monthly_co2_reduction_kg": co2_reduction,
    }
