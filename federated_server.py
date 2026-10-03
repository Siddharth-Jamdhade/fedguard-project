"""
federated_server.py — the central aggregator (runs on your laptop)

Collects weights_node_*.pkl files from each node (produced by local_model.py),
performs FedAvg (weighted average by number of local samples),
saves the global model, and can serve it back to nodes for evaluation.

Usage:
    python federated_server.py --nodes weights_node_A.pkl weights_node_B.pkl
"""

import argparse
import pickle
import numpy as np


def fedavg(node_weight_files):
    nodes = []
    for f in node_weight_files:
        with open(f, "rb") as fh:
            nodes.append(pickle.load(fh))

    total_samples = sum(n["n_samples"] for n in nodes)

    # Weighted average of coefficients and intercepts (standard FedAvg)
    coef_shape = np.array(nodes[0]["coef"]).shape
    intercept_shape = np.array(nodes[0]["intercept"]).shape

    global_coef = np.zeros(coef_shape)
    global_intercept = np.zeros(intercept_shape)

    for n in nodes:
        weight_fraction = n["n_samples"] / total_samples
        global_coef += np.array(n["coef"]) * weight_fraction
        global_intercept += np.array(n["intercept"]) * weight_fraction

    global_model = {
        "coef": global_coef.tolist(),
        "intercept": global_intercept.tolist(),
        "contributing_nodes": [n["node"] for n in nodes],
        "total_samples": total_samples,
    }

    print(f"FedAvg complete. Global model built from nodes: {global_model['contributing_nodes']}")
    print(f"NOTE: only weights were used — no raw vibration data was accessed by this script.")

    with open("global_model.pkl", "wb") as f:
        pickle.dump(global_model, f)

    return global_model


def load_global_model(global_model_path):
    """
    Reload global_model.pkl and rebuild a LogisticRegression with those
    weights, so it can be evaluated on a node's held-out local test features
    to show the global model performs well even though it never saw that
    node's raw data.
    """
    from sklearn.linear_model import LogisticRegression
    with open(global_model_path, "rb") as f:
        gm = pickle.load(f)

    model = LogisticRegression()
    model.coef_ = np.array(gm["coef"])
    model.intercept_ = np.array(gm["intercept"])
    model.classes_ = np.array([0, 1])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", required=True, help="Paths to weights_node_*.pkl files")
    args = parser.parse_args()
    fedavg(args.nodes)
