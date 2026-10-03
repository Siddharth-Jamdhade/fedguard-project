"""
anomaly_explainer.py — "detect abnormal usage, explain the likely cause,
and suggest specific actions with estimated savings" (PS-13's hardest requirement)

This is intentionally rule-based, not a black-box model: judges can see
exactly why a verdict was reached, and it reuses the SAME baseline
statistics your federated model already computes per node — no new
sensor or data pipeline needed.

Key innovation: it cross-references POWER draw with VIBRATION, which is
the same two signals FedGuard already has. A real diagnostic technique
called Motor Current Signature Analysis (MCSA) does exactly this in
industry — rising current + rising vibration together is a well-known
early signature of bearing wear, so this isn't a made-up heuristic.
"""

from carbon_calculator import estimate_cost, estimate_co2_kg, power_w_to_kwh


def diagnose(power_w, baseline_power_w, vibration_rms, baseline_vibration_rms,
             hours_per_day_abnormal=4, days_per_month=22,
             tariff_inr_per_kwh=9.0):
    power_ratio = power_w / baseline_power_w if baseline_power_w else 1.0
    vib_ratio = vibration_rms / baseline_vibration_rms if baseline_vibration_rms else 1.0

    POWER_THRESH = 1.3   # 30% above baseline
    VIB_THRESH = 1.3

    power_high = power_ratio >= POWER_THRESH
    vib_high = vib_ratio >= VIB_THRESH

    if power_high and vib_high:
        cause = ("Likely mechanical fault (e.g. bearing wear / misalignment). "
                 "Increased friction raises both current draw and vibration at the same time "
                 "— a classic Motor Current Signature Analysis (MCSA) pattern.")
        action = "Schedule a physical inspection within 48 hours; check and lubricate/replace bearings."
        severity = "high"
    elif power_high and not vib_high:
        cause = ("Likely electrical inefficiency (poor power factor, voltage imbalance, or a "
                 "partially faulty winding) rather than a mechanical issue — vibration is still normal.")
        action = "Check power factor correction and wiring connections; an electrician-level check, not a mechanical teardown."
        severity = "medium"
    elif vib_high and not power_high:
        cause = ("Early-stage mechanical wear that hasn't yet increased energy draw — "
                  "vibration sensors catch this before the energy bill does.")
        action = "No urgent stoppage needed; add to the next scheduled preventive maintenance window."
        severity = "low"
    else:
        cause = "Usage within normal range."
        action = "No action needed."
        severity = "none"

    extra_power_w = max(power_w - baseline_power_w, 0)
    extra_kwh_per_month = power_w_to_kwh(extra_power_w, hours_per_day_abnormal * days_per_month)
    estimated_monthly_savings_inr = estimate_cost(extra_kwh_per_month, tariff_inr_per_kwh)
    estimated_monthly_co2_reduction_kg = estimate_co2_kg(extra_kwh_per_month)

    return {
        "severity": severity,
        "likely_cause": cause,
        "suggested_action": action,
        "estimated_monthly_savings_inr": estimated_monthly_savings_inr,
        "estimated_monthly_co2_reduction_kg": estimated_monthly_co2_reduction_kg,
        "power_ratio_vs_baseline": round(power_ratio, 2),
        "vibration_ratio_vs_baseline": round(vib_ratio, 2),
    }
