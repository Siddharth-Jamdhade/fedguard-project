"""
update_node.py — the "live update" workflow.

Run this whenever you get a new abnormal (or updated normal) CSV for ONE node.
It retrains that node's local model, then re-runs federated averaging across
all 3 nodes, so the dashboard reflects the change on next refresh.

Usage:
    python update_node.py --node A --normal normal_A.csv --abnormal abnormal_A.csv

This assumes the OTHER two nodes' weights_node_X.pkl files already exist from
a previous run — it only retrains the node you specify, then re-aggregates
using whatever weight files are currently on disk for all 3 nodes.
"""

import argparse
import subprocess
import sys
import os

ALL_NODES = ["A", "B", "C"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, choices=ALL_NODES)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--abnormal", required=True)
    args = parser.parse_args()

    print(f"--- Retraining Node {args.node} ---")
    subprocess.run([
        sys.executable, "local_model.py",
        "--node", args.node,
        "--normal", args.normal,
        "--abnormal", args.abnormal,
    ], check=True)

    # Check all 3 nodes have weight files before re-aggregating
    missing = [n for n in ALL_NODES if not os.path.exists(f"weights_node_{n}.pkl")]
    if missing:
        print(f"\nCannot re-aggregate yet — missing weights for node(s): {missing}")
        print("Train those nodes first (each needs its own normal+abnormal CSV pair).")
        return

    print(f"\n--- Re-running federated aggregation across all nodes ---")
    weight_files = [f"weights_node_{n}.pkl" for n in ALL_NODES]
    subprocess.run([sys.executable, "federated_server.py", "--nodes"] + weight_files, check=True)

    print(f"\nDone. Node {args.node} retrained, global model refreshed.")
    print("Re-run 'streamlit run fedguard_dashboard.py' (or just refresh the browser tab if it's already running) to see the update.")


if __name__ == "__main__":
    main()
