FEDGUARD — FEDERATED PREDICTIVE MAINTENANCE — DEMO DATA
=========================================================

STORY FOR THE JUDGES:
Business A, Business B and Business C all share ONE CNC Lathe (Shared Asset #SA-07).
Each business has its OWN sensor node collecting rx/ry/rz vibration data locally.

STEP 1 — LOCAL TRAINING (raw data never leaves the business)
  business_A/company_profile.csv   -> company + employee + salary info
  business_A/sensor_data.csv       -> rx, ry, rz vibration readings (ML input), stays on Business A's node
  business_A/weights_Business_A.pkl -> ONLY the 4 numbers (3 coefficients + intercept) that leave the node
  (same structure for business_B and business_C)

  Local accuracy on their own data:
    Business A: 100.0%  (96 samples)
    Business B: 50.8%  (61 samples)   <- ambiguous data, weak local model
    Business C: 100.0%  (10 samples)  <- FAILURE PATTERN DETECTED

STEP 2 — AGGREGATION (on the FedGuard server, using federated_server.py's fedavg())
  aggregator/global_model.pkl               -> combined model, built ONLY from the 3 weight files above
  aggregator/federated_weights_summary.csv  -> human-readable table of every node's weights + the global weights

STEP 3 — GLOBAL MODEL SENT BACK TO EACH NODE
  Global model accuracy when evaluated on each node's own local data (still without touching raw data centrally):
    On Business A's data: 100.0%
    On Business B's data: 47.5%
    On Business C's data: 100.0%

KEY DEMO POINT:
No .csv sensor file or company_profile.csv ever leaves each business's own folder in the real system —
only the weights_*.pkl files (4 numbers each) travel to the aggregator. Show the judges
company_profile.csv + sensor_data.csv to prove there IS real local data, then show that
weights_*.pkl is ALL that was ever transmitted, and federated_weights_summary.csv to show
what the aggregator produced from just those numbers.
