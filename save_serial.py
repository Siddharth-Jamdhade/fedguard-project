import csv
import re
from flask import Flask, request, jsonify

app = Flask(__name__)
CSV_FILE = "vibration_data.csv"

# Write CSV header on startup
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ax", "ay", "az", "label"])

print(f"Server started! Saving incoming data to '{CSV_FILE}'...")

@app.route('/log', methods=['POST'])
def log_data():
    try:
        data = request.get_json(force=True)
        # Handle JSON input
        row = [
            data.get("timestamp", 0),
            data.get("ax", 0.0),
            data.get("ay", 0.0),
            data.get("az", 0.0),
            data.get("label", 0)
        ]
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(f"Saved row: {row}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)