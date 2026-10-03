"""
thingspeak_upload.py — Handles uploading federated metrics to ThingSpeak / Cloud telemetry.
"""

import requests

# Set your ThingSpeak Write API Key here if available, or keep default placeholder
THINGSPEAK_API_KEY = "3P6GW2FI61HRB8ZB"
THINGSPEAK_URL = "https://api.thingspeak.com/update"

def upload_federation_round(round_num, accuracy, num_nodes=3):
    """
    Uploads federated round telemetry to ThingSpeak.
    Field 1: Round Number
    Field 2: Global Accuracy
    Field 3: Active Nodes
    """
    if THINGSPEAK_API_KEY == "YOUR_API_KEY_HERE":
        print("[ThingSpeak] No active API key configured — skipping live cloud upload.")
        return False
        
    try:
        payload = {
            'api_key': THINGSPEAK_API_KEY,
            'field1': round_num,
            'field2': accuracy,
            'field3': num_nodes
        }
        response = requests.post(THINGSPEAK_URL, data=payload, timeout=5)
        if response.status_code == 200 and response.text != '0':
            print(f"[ThingSpeak] Successfully uploaded round {round_num} metrics.")
            return True
        else:
            print(f"[ThingSpeak] Upload failed or rate-limited (Response: {response.text}).")
            return False
    except Exception as e:
        print(f"[ThingSpeak] Telemetry connection error: {e}")
        return False
