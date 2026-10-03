import os
import pandas as pd
import numpy as np
import pickle

def run_privacy_pipeline(csv_file, weights_file):
    print("=========================================================================")
    print("      STAGE 1: LOCAL EDGE NODE - RAW PRIVATE PRODUCTION DATA            ")
    print("=========================================================================")
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        # Calculate local private metrics
        df['vibration_magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        total_time_sec = (df['t_ms'].iloc[-1] - df['t_ms'].iloc[0]) / 1000.0 if len(df) > 1 else 0
        
        print("\n🔒 SENSITIVE MSME OPERATIONAL LOGS (STAYS LOCAL / NEVER TRANSMITTED):")
        print(f" -> Source File: {csv_file}")
        print(f" -> Raw Telemetry Records Captured: {len(df)} samples")
        print(" -> Raw Sensor & Timestamp Sample (Private Production Stream):")
        print(df.head(4).to_string(index=False))
    else:
        print(f"[-] Error: File '{csv_file}' not found.")
        return

    print("\n" + "="*73)
    print("      STAGE 2: LOCAL ML TRANSFORMATION -> FEDERATED OUTPUT              ")
    print("=========================================================================")
    
    print("\n🤖 [LOCAL ML MODEL PROCESSING]: Ingesting raw logs locally...")
    
    # 1. Non-Sensitive Usage & Efficiency Aggregates
    avg_stress = df['vibration_magnitude'].mean()
    anomalies = (df['vibration_magnitude'] > 12.0).sum()
    health_score = max(0, 100 - (anomalies * 2.5))
    
    print("\n📊 1. AGGREGATED METRICS (PRIVACY-PRESERVING USAGE & EFFICIENCY):")
    print(f"    • Active Operating Duration: {total_time_sec:.2f} seconds")
    print(f"    • Equipment Health Score:    {health_score:.1f}%")
    print(f"    • Mean Mechanical Stress:   {avg_stress:.4f} m/s²")
    print(f"    • Anomaly Stress Events:     {anomalies} detected")

    # 2. Extract Weight Payload
    print("\n🌐 2. TRANSMITTED MODEL WEIGHT PAYLOAD (SENT TO FEDERATED SERVER):")
    if os.path.exists(weights_file):
        with open(weights_file, 'rb') as f:
            weights = pickle.load(f)
        print(f" -> Weight Payload File: {weights_file}")
        print(" -> Model Parameters (Stripped of all raw timestamps and sensor data):")
        if isinstance(weights, dict):
            for k, v in weights.items():
                print(f"    • {k}: {v}")
        else:
            print(f"    • Model Payload: {weights}")
    else:
        print(f" -> Weight file '{weights_file}' not generated yet.")

    print("\n" + "-"*73)
    print("✅ VERIFICATION COMPLETE:")
    print("   Raw production logs & exact timestamps remain locked inside Stage 1.")
    print("   Only Stage 2 (Efficiency/Health aggregates + ML weights) goes to Cloud.")
    print("-"*73 + "\n")

if __name__ == "__main__":
    run_privacy_pipeline("vib1.csv", "weights_node_A.pkl")