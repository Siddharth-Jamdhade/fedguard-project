"""
energy_federated_server.py — PS-13 FedGuard global aggregation.

Runs FedAvg (sample-weighted average) over weights from all trained nodes.
Never touches raw sensor data — only reads the weights_energy_*.pkl files.

Usage:
    python energy_federated_server.py \\
        --nodes weights_energy_A.pkl weights_energy_B.pkl weights_energy_C.pkl
"""

import argparse
import pickle
import numpy as np

from energy_storage import log_federation_round


def fedavg(node_files: list) -> dict:
    """Federated averaging: weighted mean of coefficients by n_samples."""
    nodes = []
    for f in node_files:
        with open(f, "rb") as fh:
            nodes.append(pickle.load(fh))
        print(f"  Loaded  {f}  (node={nodes[-1]['node']}, "
              f"n={nodes[-1]['n_samples']}, acc={nodes[-1]['accuracy']:.3f})")

    total = sum(n["n_samples"] for n in nodes)

    coef_shape      = np.array(nodes[0]["coef"]).shape
    intercept_shape = np.array(nodes[0]["intercept"]).shape
    mean_shape      = np.array(nodes[0]["scaler_mean"]).shape
    scale_shape     = np.array(nodes[0]["scaler_scale"]).shape

    g_coef  = np.zeros(coef_shape)
    g_inter = np.zeros(intercept_shape)
    g_mean  = np.zeros(mean_shape)
    g_scale = np.zeros(scale_shape)

    for n in nodes:
        w = n["n_samples"] / total
        g_coef  += np.array(n["coef"])         * w
        g_inter += np.array(n["intercept"])     * w
        g_mean  += np.array(n["scaler_mean"])   * w
        g_scale += np.array(n["scaler_scale"])  * w

    # Gather per-node baselines (sent back to nodes for their local comparisons)
    node_baselines = {
        n["node"]: {
            "baseline_power_w":       n["baseline_power_w"],
            "baseline_vibration_rms": n["baseline_vibration_rms"],
            "accuracy":               n["accuracy"],
            "n_samples":              n["n_samples"],
        }
        for n in nodes
    }

    global_model = {
        "coef":              g_coef.tolist(),
        "intercept":         g_inter.tolist(),
        "scaler_mean":       g_mean.tolist(),
        "scaler_scale":      g_scale.tolist(),
        "contributing_nodes": [n["node"] for n in nodes],
        "total_samples":     total,
        "node_baselines":    node_baselines,
    }

    out = "global_energy_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(global_model, f)

    log_federation_round(
        nodes=global_model["contributing_nodes"],
        total_samples=total,
    )

    print(f"\n  ✓ FedAvg complete")
    print(f"    Nodes    : {global_model['contributing_nodes']}")
    print(f"    Samples  : {total}  (weights only — no raw data accessed)")
    print(f"    Output   : {out}")
    return global_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PS-13 FedGuard: federated aggregation server")
    parser.add_argument(
        "--nodes", nargs="+", required=True,
        help="Paths to weights_energy_*.pkl from each trained node")
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"  PS-13 FedGuard — Federated Aggregation (FedAvg)")
    print(f"{'─'*60}")
    fedavg(args.nodes)
