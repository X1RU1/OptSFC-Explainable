"""
compute_silver_tree_accuracy.py
=================================
Recomputes SILVER decision tree (and logistic/linear regression, for comparison)
training accuracy directly from the saved boundary_state_discrete.csv,
using the same fitting parameters as silver_env_explain.py.

Usage:
    python compute_silver_tree_accuracy.py --input silver_envelope_env_boundary_state_discrete.csv
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


def compute_accuracy(csv_path: str):
    df = pd.read_csv(csv_path)

    # feature columns are the ones ending in "_bin"; everything else
    # (ci, cj, action) is metadata, not a model input
    feat_cols = [c for c in df.columns if c.endswith("_bin")]
    if not feat_cols:
        raise ValueError(
            "No '_bin' columns found. Make sure you passed the "
            "*_boundary_state_discrete.csv file, not the continuous one."
        )

    X = df[feat_cols].values
    y = df["action"].values
    n = len(df)

    print(f"Loaded {n} boundary points, features: {feat_cols}")
    print("\nAction distribution:")
    print(df["action"].value_counts().sort_index())
    print(f"Distinct actions present: {df['action'].nunique()} / 12 possible\n")

    # ---- Decision Tree (same params as silver_env_explain.py) ----
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X, y)
    dt_preds = dt.predict(X)
    dt_acc = (dt_preds == y).mean()
    print(f"Decision Tree accuracy:     {dt_acc:.4f}  "
          f"({(dt_preds == y).sum()}/{n})")
    print(f"  leaves = {dt.get_n_leaves()}, depth = {dt.get_depth()}")
    print(f"  feature importances: "
          f"{dict(zip(feat_cols, np.round(dt.feature_importances_, 4)))}")

    # ---- Logistic Regression ----
    if df["action"].nunique() >= 2:
        log_r = LogisticRegression(random_state=0, max_iter=10000, solver="saga")
        log_r.fit(X, y)
        log_preds = log_r.predict(X)
        log_acc = (log_preds == y).mean()
        print(f"\nLogistic Regression accuracy: {log_acc:.4f}  "
              f"({(log_preds == y).sum()}/{n})")
    else:
        print("\nLogistic Regression skipped: fewer than 2 action classes.")

    # ---- Linear Regression (round + clip), sanity check vs formulas.txt ----
    lr = LinearRegression()
    lr.fit(X, y)
    lr_raw = lr.predict(X)
    lr_preds = np.clip(np.rint(lr_raw).astype(int), y.min(), y.max())
    lr_acc = (lr_preds == y).mean()
    print(f"Linear Regression (round+clip) accuracy: {lr_acc:.4f}  "
          f"({(lr_preds == y).sum()}/{n})")

    # ---- Upper bound check: majority vote per unique discrete combo ----
    # This tells you whether the tree's accuracy is a real ceiling set by
    # the feature set, or whether the tree is underfitting.
    best_correct = 0
    for _, sub in df.groupby(feat_cols):
        best_correct += Counter(sub["action"]).most_common(1)[0][1]
    upper_bound = best_correct / n
    print(f"\nUpper bound (majority vote per unique feature combo): "
          f"{upper_bound:.4f}  ({best_correct}/{n})")
    if abs(upper_bound - dt_acc) < 1e-9:
        print("  -> Decision tree already reaches this ceiling. "
              "Not underfitting; the features themselves cannot do better.")
    else:
        print("  -> Decision tree accuracy is below the ceiling; "
              "there may be room to grow the tree further.")

    # collision detail: how many points share a discrete signature with
    # at least one other point of a different action
    grp = df.groupby(feat_cols)["action"].agg(lambda s: sorted(set(s)))
    collisions = grp[grp.apply(len) > 1]
    grp_sizes = df.groupby(feat_cols).size()
    collided_points = grp_sizes[grp.apply(len) > 1].sum()
    print(f"\nDistinct discrete feature combinations: {len(grp)}")
    print(f"Combinations with >1 distinct action: {len(collisions)}")
    print(f"Points involved in colliding combinations: {collided_points}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                         help="Path to *_boundary_state_discrete.csv")
    args = parser.parse_args()
    compute_accuracy(args.input)