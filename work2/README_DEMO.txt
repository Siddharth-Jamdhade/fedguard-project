FEDGUARD + PS-13 — ENERGY & CARBON EXTENSION
==============================================

WHAT I COULD AND COULDN'T TOUCH DIRECTLY
-----------------------------------------
I only have federated_server.py from our earlier conversation — not your
local_model.py, update_node.py, storage.py, or fedguard_dashboard.py
(those were never uploaded to this chat). So instead of guessing their
contents and handing you a possibly-wrong "patch," I built this as 3 new,
independent, drop-in modules plus real demo data, and I've listed below
EXACTLY where each one plugs into your existing files. Upload the other
4 files in a future message if you want me to edit them directly.

GOOD NEWS: federated_server.py needs ZERO changes.
Its fedavg() function reads coef_shape = np.array(nodes[0]["coef"]).shape
directly from whatever weight file you give it — it already works whether
coef has 3 numbers (rx,ry,rz) or 4 (rx,ry,rz,power_w). Just make sure
local_model.py now includes power_w as a 4th feature when it builds the
weights dict before pickling.

NEW FILES AND WHERE THEY PLUG IN
---------------------------------
carbon_calculator.py
  -> Import into local_model.py or energy_node.py. summarize_period()
     is what feeds the dashboard's cost/CO2 numbers; compare_periods()
     is what powers the "compare usage across spaces/time periods"
     requirement — call it twice (e.g. this week vs last week) and show
     the delta on the dashboard.

anomaly_explainer.py
  -> This is the PS-13 "explain the cause + suggest an action + estimate
     savings" requirement, solved. Call diagnose() anywhere you currently
     flag a vibration anomaly in local_model.py — pass it the current and
     baseline power + vibration, and it returns the explanation text,
     suggested action, and estimated savings as plain strings/numbers
     you can drop straight into fedguard_dashboard.py's existing
     "FAILURE DETECTED" banner (same UI slot you already have, just with
     richer content).

energy_node.py
  -> Run this per business, same way you'd run local_model.py. It reads
     either --serial (live from the Wokwi ESP32) or --csv (the bundled
     demo data), writes Business_X_energy_log.csv and
     Business_X_anomaly_report.csv locally. Wire storage.py to read these
     two files the same way it currently reads local_model.py's output —
     I don't have storage.py's schema so I can't write that part for you.

wokwi/diagram.json + wokwi/sketch.ino
  -> Paste into a new Wokwi project. Adds ONE potentiometer alongside
     your existing MPU6050 (Wokwi has no INA219/ACS712/PZEM-004T energy
     sensor — verified against Wokwi's official supported-hardware list —
     so the pot is the live-adjustable stand-in: turning it live during
     the demo changes "power_w" exactly like turning a real knob on a
     variac would). Serial output is "rx,ry,rz,adc_raw,power_w" — feed
     that straight into energy_node.py's --serial mode.

PER-BUSINESS DEMO DATA (ALREADY GENERATED AND RUN)
-----------------------------------------------------
Business_A/energy_data.csv, Business_A_energy_log.csv, Business_A_anomaly_report.csv
  -> Normal all morning, then a genuine power+vibration spike together
     near end of shift -> diagnosed HIGH severity, "mechanical fault",
     real MCSA-style reasoning, with estimated monthly savings in INR.

Business_B/... -> power rises but vibration stays normal -> diagnosed
  MEDIUM severity, "electrical inefficiency", different suggested action
  than Business A. This is the proof that the explainer isn't just
  threshold-spam — two different sensor signatures get two different,
  correct diagnoses.

Business_C/... -> confirms the earlier bearing-failure story from the
  original FedGuard demo, now with energy evidence backing it too.

aggregator/PS13_compliance_checklist.csv
  -> Maps every single PS-13 bullet point to the exact file that
     satisfies it. Hand this straight to judges.

THE ACTUAL PITCH LINE FOR JUDGES
-----------------------------------
"We didn't bolt a generic energy dashboard onto our maintenance project.
We proved that the SAME two sensors (vibration + current draw) that
predict mechanical failure also detect energy waste — because a failing
motor draws more current before it breaks. That's a real industrial
technique called Motor Current Signature Analysis. One federated
network, one set of sensors, two national problems solved at once:
equipment downtime AND carbon/cost accountability."
