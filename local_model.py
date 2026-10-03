"""
local_model.py — run once per node (Node A, Node B, Node C)
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from storage import log_local_round

WINDOW_SIZE = 5


def extract_features(df):
    if "mag" not in df.columns:
        df["mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    
    features = []
    if len(df) <= WINDOW_SIZE:
        window = df["mag"]
        std_val = window.std() if len(window) > 1 else 0.0
        features.append([window.mean(), std_val, window.max() - window.min()])
        return np.array(features)

    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df["mag"].iloc[i:i + WINDOW_SIZE]
        features.append([
            window.mean(),
            window.std() if len(window) > 1 else 0.0,
            window.max() - window.min(),
        ])
    return np.array(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, help="Node identifier, e.g. A")
    parser.add_argument("--normal", required=True, help="CSV of normal vibration data")
    parser.add_argument("--abnormal", required=True, help="CSV of abnormal vibration data")
    args = parser.parse_args()

    normal_df = pd.read_csv(args.normal)
    abnormal_df = pd.read_csv(args.abnormal)

    X_normal = extract_features(normal_df)
    X_abnormal = extract_features(abnormal_df)

    X = np.vstack([X_normal, X_abnormal])
    y = np.array([0] * len(X_normal) + [1] * len(X_abnormal))

    test_sz = 0.25 if len(X) >= 8 else 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Node {args.node} local model accuracy: {acc:.2f}")

    # Log to local_history table in storage.py
    log_local_round(node=args.node, accuracy=acc, n_samples=len(X_train))

    weights = {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "node": args.node,
        "n_samples": len(X_train),
    }
    with open(f"weights_node_{args.node}.pkl", "wb") as f:
        pickle.dump(weights, f)

    print(f"Saved weights_node_{args.node}.pkl — send this to the federated aggregator.")


if __name__ == "__main__":
    main()
