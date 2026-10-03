import csv
import sys

# Ask which MSME node file you are logging
filename = input("Enter output CSV filename (e.g., node2.csv or node3.csv): ").strip()
if not filename.endswith(".csv"):
    filename += ".csv"

# Create/Overwrite file with standard header
with open(filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["t_ms", "ax", "ay", "az"])

print(f"\n[+] Saving data directly to '{filename}'")
print("Paste your copied Serial Monitor text below, then press Enter:")

try:
    for line in sys.stdin:
        line = line.strip()
        # Filter out headers/connection messages and keep CSV rows
        if line and not line.startswith("t_ms") and not line.startswith("Connect"):
            parts = line.split(",")
            if len(parts) == 4:
                with open(filename, mode='a', newline='') as f:
                    f.write(line + "\n")
                print(f"Recorded: {line}")
except KeyboardInterrupt:
    print(f"\n[✓] Data successfully saved to {filename}!")