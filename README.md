# FedGuard — PS-13 Smart Energy Consumption & Carbon Footprint Monitoring

> **Hackathon Project · Problem Statement PS-13**
> Institutions, offices and households often lack visibility into energy usage, cost and environmental impact.

---

## 🧠 What FedGuard Does

FedGuard is a **privacy-preserving federated learning system** that monitors energy consumption across multiple industrial nodes (factories, offices, workshops), estimates real-time cost and CO₂ emissions, detects anomalies automatically, explains their likely cause (Motor Current Signature Analysis), and suggests corrective actions with estimated savings — all **without sharing raw sensor data** between nodes.

---

## 🏗️ System Architecture

```
                         ┌─────────────────────────────────┐
                         │        Wokwi Simulation          │
                         │  ESP32 + MPU6050 + Potentiometer │
                         │  Tab A (Factory) / B (Office) /  │
                         │  C (Workshop)                    │
                         └──────────────┬──────────────────┘
                                        │ CSV: t_ms, power_w,
                                        │      ax, ay, az, vibration_rms
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
           energy_A.csv         energy_B.csv         energy_C.csv
                    │                   │                    │
                    ▼                   ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ Local Model A    │ │ Local Model B    │ │ Local Model C    │
        │ LogisticRegress. │ │ LogisticRegress. │ │ LogisticRegress. │
        │ + StandardScaler │ │ + StandardScaler │ │ + StandardScaler │
        └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                 │  weights_energy_A   │  weights_energy_B   │  weights_energy_C
                 └───────────┬─────────┘─────────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │  FedAvg Aggregator   │
                  │  (No raw data shared)│
                  │  global_energy_model │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Streamlit Dashboard │
                  │  4 tabs: Overview /  │
                  │  Comparison /        │
                  │  Anomaly Detection / │
                  │  Federated Learning  │
                  └──────────────────────┘
```

---

## ⚡ Key Features

| Feature | Description |
|---|---|
| **Energy Monitoring** | Real-time power (W) → kWh → ₹ cost → kg CO₂ |
| **3-Node Simulation** | Factory · Office · Workshop via Wokwi ESP32 tabs |
| **Anomaly Detection** | MCSA pattern matching — detects mechanical faults, bearing wear, PF degradation |
| **Explainability** | Every alert includes: likely cause + corrective action + estimated ₹ savings/month |
| **Federated Learning** | FedAvg — models trained locally, only weights aggregated |
| **Privacy Preserving** | Raw sensor data never leaves each node |
| **India-Specific** | Tariff ₹8/kWh · CEA emission factor 0.82 kg CO₂/kWh |

---

## 📁 Project Structure

```
FedGuard/
│
├── wokwi_energy/
│   ├── diagram.json          # Wokwi circuit (ESP32 + MPU6050 + Pot + A4988)
│   └── sketch.ino            # ESP32 firmware → CSV serial output
│
├── generate_energy_data.py   # Generate synthetic demo CSVs
├── energy_local_model.py     # Per-node local training (LogisticRegression)
├── energy_federated_server.py# FedAvg aggregation across nodes
├── energy_dashboard.py       # Streamlit dashboard (4 tabs)
├── energy_carbon.py          # Cost (₹) + CO₂ estimation utilities
├── energy_anomaly.py         # MCSA anomaly diagnosis engine
├── energy_storage.py         # SQLite audit trail
│
├── energy_A.csv              # Sample data — Business A (Factory)
├── energy_B.csv              # Sample data — Business B (Office)
├── energy_C.csv              # Sample data — Business C (Workshop)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quickstart (5 minutes)

### Prerequisites
- Python 3.10+ (tested on 3.13)
- pip

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate sample data
```bash
python generate_energy_data.py
```
> Creates `energy_A.csv`, `energy_B.csv`, `energy_C.csv`

### 3. Train local models on each node
```bash
python energy_local_model.py --node A --csv energy_A.csv
python energy_local_model.py --node B --csv energy_B.csv
python energy_local_model.py --node C --csv energy_C.csv
```

### 4. Federated aggregation
```bash
python energy_federated_server.py --nodes weights_energy_A.pkl weights_energy_B.pkl weights_energy_C.pkl
```

### 5. Launch dashboard
```bash
streamlit run energy_dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## 🔌 Wokwi Hardware Simulation

**Open:** [wokwi.com](https://wokwi.com) → New Project → ESP32

Paste the contents of `wokwi_energy/diagram.json` and `wokwi_energy/sketch.ino`.

### Circuit Summary

| Component | Pin |
|---|---|
| MPU6050 SDA | GPIO 21 |
| MPU6050 SCL | GPIO 22 |
| Potentiometer SIG | GPIO 34 (ADC) |
| A4988 STEP | GPIO 16 |
| A4988 DIR | GPIO 17 |

### Serial Output Format
```
t_ms,power_w,ax,ay,az,vibration_rms
1000,654.3,0.1230,0.0980,9.8012,9.8017
```

### Demo Instructions (3 Tabs)

| Tab | Node | Pot Setting | Anomaly Demo |
|---|---|---|---|
| Tab A | Factory | 60–75% (600–800 W) | Turn to 95%+ after 30 readings |
| Tab B | Office | 20–30% (200–350 W) | Turn to 60%+ after 30 readings |
| Tab C | Workshop | 40–55% (400–550 W) | Gradual increase |

---

## 🤖 ML Model Details

### Algorithm: Logistic Regression (FedAvg-compatible)
- **Why LR?** FedAvg requires averaging numeric weight vectors — LR's `coef_` and `intercept_` can be directly averaged across nodes without a FL framework
- **Features (8 per window):** `mean_power`, `std_power`, `max_power`, `power_range`, `power_slope`, `mean_vibration`, `std_vibration`, `max_vibration`
- **Labels:** Auto-generated from first 30% baseline — power >1.35× OR vibration >1.40× = anomaly
- **Scaler:** StandardScaler parameters also federated (mean + scale averaged)

### MCSA Rule Engine (Anomaly Explainer)

| Power Ratio | Vibration Ratio | Diagnosis |
|---|---|---|
| ≥1.50 | ≥1.80 | 🔴 Mechanical fault — bearing replacement needed |
| ≥1.25 | <1.30 | 🟡 Power factor degradation — capacitor bank check |
| <1.25 | ≥1.40 | 🟡 Early bearing wear — schedule lubrication |
| <1.10 | <1.15 | ✅ Normal operation |

---

## 📊 Estimation Methodology

| Metric | Formula | Source |
|---|---|---|
| Power (W) | `ADC / 4095 × 2500` | Potentiometer linear map |
| Energy (kWh) | `Σ power_w × interval_hours / 1000` | Accumulated |
| Cost (₹) | `energy_kwh × 8.0` | India MSME industrial tariff |
| CO₂ (kg) | `energy_kwh × 0.82` | CEA Grid Emission Factor 2023 |
| Monthly savings | `excess_w × 200 hrs/month / 1000 × 8.0` | 25 days × 8 hr/day |

---

## 🌍 Why Federated Learning?

In a real MSME deployment:
- **Business A** (factory) won't share production data with **Business B** (competitor office)
- Federated Learning lets all 3 nodes train a **shared global model** without exchanging raw sensor readings
- Only model weights (small numeric arrays) are sent to the aggregator
- Each node's data stays on-premises

```
Privacy guarantee:
  Raw data:   stays at each node      ✅
  What moves: 8-element weight vector ✅
  Server sees: aggregated weights only ✅
```

---

## 🛠️ Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key settings:
```
TARIFF_INR_PER_KWH=8.0          # Your state DISCOM rate
EMISSION_FACTOR_KG_CO2_PER_KWH=0.82  # CEA 2023
SERIAL_PORT=COM3                 # For live ESP32 data
```

---

## 📋 Requirements

```
scikit-learn==1.5.2
numpy==2.1.1
pandas==2.2.3
streamlit==1.39.0
plotly==5.24.1
pyserial==3.5
joblib==1.4.2
```

---

## 👥 Team

| Name | Role |
|---|---|
| Siddharth Jamdhade | Lead Developer · ML · Dashboard |

---

## 📄 License

MIT License — free to use, modify and distribute.

---

*Built for hackathon PS-13 — Smart Energy Consumption & Carbon Footprint Monitoring System*
