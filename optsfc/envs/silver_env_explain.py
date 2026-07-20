"""
SILVER with RL-Guided Labeling (env_action variant)
=====================================================
Builds a global interpretable surrogate policy from the Shapley vectors
(Φ_s) produced by shap_env_explain.py.

Pipeline (paper-aligned, env_action variant)
---------------------------------------------
Step 1  Load Φ_s and original state features X from shap_env_explain output
Step 2  KMeans on Φ_s  →  cluster labels + centroids
Step 3  Find boundary points in Φ_s space (one per cluster pair)
Step 4  Inverse mapping: boundary Φ_s  →  original state X
Step 5  RL-Guided Labeling: read action label directly from env_action column
            ALL algos (DQN / Envelope / PPO / A2C / EUPG) → env_action
            This is the action the agent actually executed in the environment,
            which may differ from argmax(Q/prob) for stochastic policies.
Step 5b Quantile discretization of bd_state (Plan B)
            Compute per-feature tertile edges from the full trajectory data X
            (not from boundary points) so the bins reflect the true feature
            distribution.  Each continuous value is mapped to an integer bin:
                0  →  low    (below 33rd percentile of X)
                1  →  medium (33rd–66th percentile of X)
                2  →  high   (above 66th percentile of X)
            The same bin_edges dict is applied to bd_state before surrogate
            model training, and must be reused in the APG stage so that
            tree.apply() receives identically discretized features.
Step 6  Fit interpretable surrogate models on (bd_state_discrete, action labels):
            Decision Tree   (primary, fully interpretable)
            Linear Regression   (continuous approximation, round+clip)
            Logistic Regression (probabilistic multi-class)
Step 7  Save models, boundary datasets, bin_edges, and formula strings

Discretization design note
---------------------------
bin_edges is computed from the full per-algo trajectory X, not from the
66 boundary points.  This ensures the low/medium/high thresholds reflect
realistic feature ranges rather than the narrow boundary subset.
bin_edges is saved alongside the decision tree so the APG stage can apply
the identical mapping to raw trajectory features before calling tree.apply().

Display note (decision tree)
-----------------------------
The decision tree is TRAINED on discretized integer bins (0/1/2/...), so
its raw split thresholds are bin-index cut points (e.g. 0.5, 1.5), not
real feature values.  For human-readable output (the PDF plot and the
"Decision Tree" section of formulas.txt), a display copy of the tree is
built with _tree_with_real_thresholds(), which rewrites each split
threshold back into the corresponding real value from bin_edges.  The
pickled tree used for prediction / apply() is never touched by this.

APG alignment guarantee
------------------------
When assigning leaf nodes in the APG stage:
    X_discrete = discretize(X_traj, bin_edges)
    leaf_ids   = dt.apply(X_discrete)
Both the tree and the input share the same discrete feature space, so
leaf-node membership is semantically consistent with the surrogate's training.

Consistency guarantee
----------------------
Φ_s files loaded here (shap_*_env.csv) were produced by
shap_env_explain.py, which sliced the SHAP tensor at env_action.
The action labels in Step 5 are also read from env_action.
Both therefore refer to the SAME executed action at every timestep.

Difference from silver_explain.py (argmax variant)
----------------------------------------------------
- SHAP files: shap_*_env.csv  (from shap_env_outputs/, no subdirectories)
- Action labels: env_action column, NOT argmax(Q/prob)
- Output directory: silver_env_outputs/ by default
- Output filenames: silver_<tag>_env_*.pkl / *.csv / *.txt

Input files (from shap_env_explain.py output dir)
---------------------------------------------------------
    shap_env_outputs/shap_<tag>_env.csv   — Φ_s matrix, shape (N, n_features)
    No subdirectory structure; all files are flat under shap_env_outputs/.

Input files (original data CSV)
--------------------------------
    --input CSV    — must contain FEATURE_COLS + env_action column

Output files (per algo)
------------------------
    silver_<tag>_env_kmeans.pkl
    silver_<tag>_env_boundary_shap.csv
    silver_<tag>_env_boundary_state.csv          (continuous, for reference)
    silver_<tag>_env_boundary_state_discrete.csv (discretized, used for training)
    silver_<tag>_env_bin_edges.pkl               (bin_edges dict; reuse in APG)
    silver_<tag>_env_decision_tree.pkl
    silver_<tag>_env_linear_regression.pkl
    silver_<tag>_env_logistic_regression.pkl
    silver_<tag>_env_formulas.txt
    silver_<tag>_env_decision_tree.pdf

Usage
-----
    python silver_env_explain.py --input data.csv --shap_dir shap_env_outputs --output silver_env_outputs
    python silver_env_explain.py --input data.csv --shap_dir shap_env_outputs --output silver_env_outputs --algo DQN
    python silver_env_explain.py --input data.csv --shap_dir shap_env_outputs --output silver_env_outputs --n_bins 3
"""

import os
import copy
import pickle
import warnings
import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants — must match shap_env_explain.py
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "feat_mean_mtd_overhead",
    "feat_mean_network_penalty",
    "feat_max_network_penalty",
    "feat_mean_security_penalty",
    "feat_max_security_penalty",
]

# FEATURE_COLS = [
#     # --- Security ---
#     "feat_max_apt_score",           # apt cvss/asp score 
#     "feat_mean_apt_score",
#     "feat_max_dataleak_score",      # data_leak cvss/asp score 
#     "feat_mean_dataleak_score",
#     "feat_max_dos_score",           # dos cvss/asp score 
#     "feat_mean_dos_score",

#     # --- Resource ---
#     "feat_vim0_cpu",               
#     "feat_vim0_ram",
#     "feat_vim1_cpu",
#     "feat_vim1_ram",
#     "feat_mean_remaining_mig",
#     "feat_mean_remaining_reinst",

#     # --- Network ---
#     "feat_total_ues",              
# ]

N_ACTIONS      = 12
ENV_ACTION_COL = "env_action"

# Default number of quantile bins per feature (3 → low / medium / high).
# Increasing this (e.g. to 4 or 5) produces finer-grained leaf conditions
# at the cost of a larger discrete state space.
DEFAULT_N_BINS = 3

# Human-readable bin labels used in formula output and tree annotations.
# Length must equal DEFAULT_N_BINS (or the --n_bins argument).
BIN_LABELS = ["low", "medium", "high"]

# Number of decimal places used when rendering real-valued thresholds in
# the tree plot (PDF) and in export_text() for formulas.txt.
TREE_DISPLAY_PRECISION = 4

# Φ_s files produced by shap_env_explain.py.
# All files are flat under shap_dir/ (no subdirectories).
# _env suffix distinguishes them from the argmax variant.
SHAP_FILE = {
    "DQN":      "shap_dqn_scalar_Q_env.csv",
    "PPO":      "shap_ppo_policy_prob_env.csv",
    "A2C":      "shap_a2c_policy_prob_env.csv",
    "EUPG":     "shap_eupg_policy_prob_env.csv",
    "Envelope": "shap_envelope_scalar_Q_env.csv",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_state_features(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLS].values.astype(np.float64)


def _load_shap(algo: str, shap_dir: str) -> np.ndarray:
    """
    Load Φ_s matrix for algo from shap_dir (flat, no subdirectories).

    Files were produced by shap_env_explain.py and named
    shap_*_env.csv.  The Φ_s in each file was sliced at env_action,
    matching the action labels produced by _get_action_labels below.
    """
    fname = SHAP_FILE[algo]
    p     = os.path.join(shap_dir, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing SHAP file: {p}\n"
            f"Expected shap_env_explain.py output at: {shap_dir}/"
        )
    return pd.read_csv(p)[FEATURE_COLS].values.astype(np.float64)


def _get_action_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Read the action label for every state from the env_action column.

    This is the action the agent actually executed in the environment.
    All five algorithms use the same column — no per-algo branching needed.
    Mirrors the env_action slice used in shap_env_explain.py so
    Φ_s and the action label always refer to the same executed action.

    Raises
    ------
    ValueError  if env_action column is absent or contains out-of-range values.
    """
    if ENV_ACTION_COL not in df.columns:
        raise ValueError(
            f"Column '{ENV_ACTION_COL}' not found in input CSV. "
            "The env_action variant requires the actual executed action index."
        )
    actions = df[ENV_ACTION_COL].values.astype(int)
    if actions.min() < 0 or actions.max() >= N_ACTIONS:
        raise ValueError(
            f"'{ENV_ACTION_COL}' values must be in [0, {N_ACTIONS - 1}]. "
            f"Got range [{actions.min()}, {actions.max()}]."
        )
    return actions


# ---------------------------------------------------------------------------
# Step 2: KMeans on Φ_s
# ---------------------------------------------------------------------------

def _kmeans(shap_data: np.ndarray, n_clusters: int, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(shap_data)
    return km


# ---------------------------------------------------------------------------
# Step 3: Boundary point identification
# ---------------------------------------------------------------------------

def _find_boundary_points(shap_data: np.ndarray, centroids: np.ndarray):
    """
    For every pair of clusters (i, j), find the sample in Φ_s space
    whose distance to centroid_i and centroid_j is most equal
    (i.e. closest to the decision boundary between i and j).

    Returns list of (shap_vector, original_row_idx, (cluster_i, cluster_j)).
    """
    n_clusters      = len(centroids)
    boundary_points = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dists     = cdist(shap_data, [centroids[i], centroids[j]], "euclidean")
            dist_diff = np.abs(dists[:, 0] - dists[:, 1])
            idx       = np.argmin(dist_diff)
            boundary_points.append((shap_data[idx], idx, (i, j)))

    return boundary_points


# ---------------------------------------------------------------------------
# Step 4 + 5: Inverse mapping + RL-Guided Labeling (env_action)
# ---------------------------------------------------------------------------

def _build_boundary_dataset(
    boundary_points: list,
    X: np.ndarray,
    shap_data: np.ndarray,
    action_labels: np.ndarray,
):
    """
    For each boundary point:
        - Φ_s vector  (from step 3)
        - original state X[row_idx]  (inverse mapping via stored row index)
        - action label from env_action  (RL-guided labeling)

    Returns
    -------
    bd_shap  : (n_pairs, n_features) — boundary in Φ_s space
    bd_state : (n_pairs, n_features) — boundary in state space (continuous)
    bd_y     : (n_pairs,)            — env_action labels
    meta     : list of (ci, cj) tuples
    """
    bd_shap, bd_state, bd_y, meta = [], [], [], []

    for shap_vec, row_idx, (ci, cj) in boundary_points:
        bd_shap.append(shap_vec)
        bd_state.append(X[row_idx])
        bd_y.append(action_labels[row_idx])
        meta.append((ci, cj))

    return (
        np.array(bd_shap),
        np.array(bd_state),
        np.array(bd_y),
        meta,
    )


# ---------------------------------------------------------------------------
# Step 5b: Quantile discretization (Plan B)
# ---------------------------------------------------------------------------

def compute_bin_edges(X: np.ndarray, feature_names: list, n_bins: int = DEFAULT_N_BINS) -> dict:
    """
    Compute per-feature quantile bin edges from the full trajectory data X.

    Edges are computed from the complete per-algo trajectory (not from the
    66 boundary points) so the low/medium/high thresholds reflect the true
    distribution of each feature across all observed states.

    Parameters
    ----------
    X            : (N, n_features) continuous feature matrix — full trajectory
    feature_names: list of feature name strings (length must equal X.shape[1])
    n_bins       : number of equal-frequency bins (default 3 → low/medium/high)

    Returns
    -------
    bin_edges : dict mapping feature_name → 1-D array of (n_bins - 1) cut points
                Example for n_bins=3:
                    {"feat_mean_mtd_overhead": [0.23, 0.67], ...}
                A value v is assigned bin 0 if v <= edges[0],
                bin 1 if edges[0] < v <= edges[1], ..., bin n_bins-1 otherwise.
    """
    bin_edges = {}
    quantile_steps = [i / n_bins for i in range(1, n_bins)]   # e.g. [0.333, 0.667]
    for col_idx, name in enumerate(feature_names):
        col    = X[:, col_idx]
        edges  = np.quantile(col, quantile_steps)
        bin_edges[name] = edges
    return bin_edges


def discretize(X: np.ndarray, bin_edges: dict, feature_names: list) -> np.ndarray:
    """
    Map a continuous feature matrix to integer bin indices using precomputed edges.

    Each value v in feature column f is assigned:
        bin 0  (low)    if v <= bin_edges[f][0]
        bin 1  (medium) if bin_edges[f][0] < v <= bin_edges[f][1]
        ...
        bin k  (high)   if v > bin_edges[f][k-1]

    This function is the single source of truth for the low/medium/high
    mapping.  Call it identically in both the SILVER training stage
    (on bd_state) and the APG assignment stage (on trajectory X) so that
    the decision tree always receives the same discrete representation.

    Parameters
    ----------
    X            : (N, n_features) continuous feature matrix
    bin_edges    : dict returned by compute_bin_edges()
    feature_names: list of feature name strings matching X columns

    Returns
    -------
    X_discrete : (N, n_features) integer array of bin indices (dtype int32)
    """
    X_discrete = np.zeros_like(X, dtype=np.int32)
    for col_idx, name in enumerate(feature_names):
        edges              = bin_edges[name]                    # shape (n_bins-1,)
        X_discrete[:, col_idx] = np.digitize(X[:, col_idx], edges, right=True)
        # np.digitize with right=True returns:
        #   0  if v <= edges[0]   →  low
        #   1  if edges[0] < v <= edges[1]  →  medium   (for n_bins=3)
        #   2  if v > edges[1]   →  high
    return X_discrete


def _bin_edges_summary(bin_edges: dict, feature_names: list,
                       n_bins: int, labels: list) -> str:
    """
    Build a human-readable summary of the discretization thresholds.

    Example output (n_bins=3):
        feat_mean_mtd_overhead    : (-inf, 0.2314] → low
                                    (0.2314, 0.6821] → medium
                                    (0.6821, +inf) → high
        ...
    """
    lines = [f"Discretization scheme: {n_bins}-bin quantile  "
             f"({' / '.join(labels)})\n"]
    for name in feature_names:
        edges = bin_edges[name]
        lines.append(f"  {name}:")
        thresholds = [-np.inf] + list(edges) + [np.inf]
        for k in range(len(thresholds) - 1):
            lo    = thresholds[k]
            hi    = thresholds[k + 1]
            label = labels[k] if k < len(labels) else str(k)
            lo_str = f"{lo:.4f}" if np.isfinite(lo) else "-inf"
            hi_str = f"{hi:.4f}" if np.isfinite(hi) else "+inf"
            lines.append(f"    ({lo_str}, {hi_str}]  →  {label}  (bin {k})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree display helper: bin-index thresholds → real feature values
# ---------------------------------------------------------------------------

def _tree_with_real_thresholds(dt: DecisionTreeClassifier, feature_names: list,
                               bin_edges: dict) -> DecisionTreeClassifier:
    """
    Return a DEEP COPY of dt whose internal split thresholds have been
    rewritten from discretized bin-index space (e.g. 0.5, 1.5) into the
    real continuous feature values recorded in bin_edges.

    Why this mapping is correct
    ----------------------------
    dt was fit on integer bin indices (0=low, 1=medium, 2=high, ...).
    At any internal node, sklearn picks a split threshold t such that
    "go left if value <= t, go right if value > t".  Since the feature
    values seen during training are always integers in [0, n_bins-1],
    t always falls in the gap between two of them, so floor(t) identifies
    exactly which bin edge the split corresponds to:
        floor(t) = k   ⇔   split separates bin <= k from bin > k
                        ⇔   real-world cutoff is bin_edges[feature][k]
    This holds for the usual case (t = k + 0.5) and also for the edge
    case where an intermediate bin is missing from the training subset
    at that node (t can then land on an integer, but floor(t) still
    gives the correct k).

    This copy is for PLOTTING / TEXT EXPORT ONLY. Never use it for
    prediction or apply() — its thresholds no longer live in the
    discretized space the tree was actually trained on.
    """
    dt_display = copy.deepcopy(dt)
    tree_ = dt_display.tree_
    for node_id in range(tree_.node_count):
        feat_idx = tree_.feature[node_id]
        if feat_idx < 0:
            continue  # leaf node, nothing to rewrite
        feat_name = feature_names[feat_idx]
        edges     = bin_edges[feat_name]           # shape (n_bins - 1,)
        k         = int(np.floor(tree_.threshold[node_id]))
        k         = max(0, min(k, len(edges) - 1))  # clamp defensively
        tree_.threshold[node_id] = edges[k]
    return dt_display


# ---------------------------------------------------------------------------
# Step 6: Surrogate models
# ---------------------------------------------------------------------------

def _fit_decision_tree(bd_X: np.ndarray, bd_y: np.ndarray) -> DecisionTreeClassifier:
    """
    Fit a decision tree on the discretized boundary state features.

    The tree is trained without constraining max_depth so that it can
    fully capture the action boundaries in the discretized feature space.
    Because the input is already discretized (integer bins), split
    thresholds are in bin-index space.  For display, use
    _tree_with_real_thresholds() to convert them back to real values.
    """
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(bd_X, bd_y)
    return dt


def _fit_linear_regression(bd_X: np.ndarray, bd_y: np.ndarray) -> LinearRegression:
    lr = LinearRegression()
    lr.fit(bd_X, bd_y)
    return lr


def _fit_logistic_regression(bd_X: np.ndarray, bd_y: np.ndarray) -> LogisticRegression:
    log_r = LogisticRegression(
        random_state=0, max_iter=10000, solver="saga", multi_class="multinomial"
    )
    log_r.fit(bd_X, bd_y)
    return log_r


def _round_and_clip(predictions: np.ndarray, min_val: int = 0, max_val: int = None):
    if max_val is None:
        max_val = N_ACTIONS - 1
    return np.clip(np.rint(predictions).astype(int), min_val, max_val)


# ---------------------------------------------------------------------------
# Step 7: Formula strings
# ---------------------------------------------------------------------------

def _linear_formula(lr: LinearRegression, feature_names: list,
                    fname: str = "f", tol: float = 1e-16) -> str:
    coefs     = lr.coef_.ravel()
    intercept = float(lr.intercept_)
    terms     = []
    for w, name in zip(coefs, feature_names):
        if abs(w) < tol:
            continue
        sign = "+" if w > 0 else "-"
        terms.append(f"{sign} {abs(w):.4e}·{name}")
    body = " ".join(terms) if terms else ""
    return f"{fname}(x) = {intercept:.4e} {body}"


def _logistic_formulas(log_r: LogisticRegression, feature_names: list,
                       class_names: list, fname: str = "f") -> str:
    coef      = log_r.coef_
    intercept = log_r.intercept_
    if coef.shape[0] == 1 and len(class_names) == 2:
        coef      = np.vstack([-coef[0], coef[0]])
        intercept = np.array([-intercept[0], intercept[0]])
    lines = []
    for k, cls in enumerate(class_names):
        if k >= len(coef):
            break
        terms = []
        for c, name in zip(coef[k], feature_names):
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {abs(c):.4e}*{name}")
        lines.append(f"{fname}_{cls}(x) = {intercept[k]:.4e} " + " ".join(terms))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def _save_csv(df: pd.DataFrame, path: str):
    """
    Save an already-assembled DataFrame (feature/shap columns plus any
    extra metadata columns such as ci/cj/action) to CSV.
    """
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


def _save_tree_plot(dt: DecisionTreeClassifier, feature_names: list,
                    class_names: list, title: str, path: str,
                    bin_edges: dict):
    """
    Render the decision tree to a PDF with REAL feature-value thresholds.

    dt was trained on discretized (bin-index) features, so its raw
    thresholds are bin cut points, not real values.  Before plotting we
    build a display-only copy via _tree_with_real_thresholds() that
    rewrites every split threshold into the corresponding real value
    from bin_edges, so the rendered tree is directly human-readable
    without needing to cross-reference the discretization scheme.
    """
    dt_plot = _tree_with_real_thresholds(dt, feature_names, bin_edges)
    fig, ax = plt.subplots(figsize=(max(20, len(class_names) * 4), 12))
    plot_tree(dt_plot, feature_names=feature_names, class_names=class_names,
              filled=True, impurity=False, ax=ax,
              precision=TREE_DISPLAY_PRECISION)
    ax.set_title(title, fontsize=16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Per-algo SILVER runner (env_action variant)
# ---------------------------------------------------------------------------

def run_silver_env(
    algo: str,
    df: pd.DataFrame,
    shap_dir: str,
    output_dir: str,
    n_clusters: int = None,
    n_bins: int = DEFAULT_N_BINS,
):
    """
    Full SILVER pipeline for one algorithm (env_action variant, Plan-B
    quantile discretization).

    Parameters
    ----------
    algo        : one of DQN / Envelope / PPO / A2C / EUPG
    df          : original data DataFrame (must contain env_action column)
    shap_dir    : directory containing shap_env_explain.py output CSVs
                  (flat structure, no subdirectories)
    output_dir  : where to write silver_*_env outputs
    n_clusters  : KMeans k  (default: N_ACTIONS = 12)
    n_bins      : number of quantile bins for discretization
                  (default 3 → low / medium / high)
    """
    if n_clusters is None:
        n_clusters = N_ACTIONS

    tag = algo.lower() if algo != "Envelope" else "envelope"
    print(f"\n{'='*60}\nSILVER (env_action): {algo}  "
          f"(k={n_clusters}, n_bins={n_bins})\n{'='*60}")

    # ── Step 1: load Φ_s and state features ──────────────────────────────
    shap_data     = _load_shap(algo, shap_dir)      # (N, 5)
    X             = _load_state_features(df)         # (N, 5) — full trajectory
    action_labels = _get_action_labels(df)           # (N,) — from env_action column
    N             = len(shap_data)
    print(f"  Loaded {N} samples,  Φ_s shape={shap_data.shape}")
    print(f"  Action labels: env_action  "
          f"(unique={sorted(set(action_labels.tolist()))})")

    # ── Step 2: KMeans on Φ_s ────────────────────────────────────────────
    km        = _kmeans(shap_data, n_clusters)
    centroids = km.cluster_centers_
    _save_pkl(km, os.path.join(output_dir, f"silver_{tag}_env_kmeans.pkl"))
    print(f"  KMeans done — {n_clusters} clusters")

    # ── Step 3: boundary points ───────────────────────────────────────────
    boundary_points = _find_boundary_points(shap_data, centroids)
    n_pairs         = len(boundary_points)
    print(f"  Found {n_pairs} boundary points  (C({n_clusters},2)={n_pairs})")

    # ── Step 4+5: inverse mapping + env_action labeling ──────────────────
    bd_shap, bd_state, bd_y, meta = _build_boundary_dataset(
        boundary_points, X, shap_data, action_labels
    )

    # save boundary datasets (continuous values kept for reference)
    shap_cols  = [f"shap_{f}" for f in FEATURE_COLS]
    state_cols = list(FEATURE_COLS)
    ci_col     = [m[0] for m in meta]
    cj_col     = [m[1] for m in meta]

    bd_shap_df           = pd.DataFrame(bd_shap,  columns=shap_cols)
    bd_shap_df["ci"]     = ci_col
    bd_shap_df["cj"]     = cj_col
    bd_shap_df["action"] = bd_y
    _save_csv(
        bd_shap_df,
        os.path.join(output_dir, f"silver_{tag}_env_boundary_shap.csv"),
    )

    # continuous boundary state — saved for reference / debugging only;
    # surrogate models are trained on the discretized version below
    bd_state_df           = pd.DataFrame(bd_state, columns=state_cols)
    bd_state_df["ci"]     = ci_col
    bd_state_df["cj"]     = cj_col
    bd_state_df["action"] = bd_y
    _save_csv(
        bd_state_df,
        os.path.join(output_dir, f"silver_{tag}_env_boundary_state.csv"),
    )
    print(f"  Boundary datasets saved  (n={len(bd_y)}, "
          f"unique actions={sorted(set(bd_y.tolist()))})")

    # ── Step 5b: quantile discretization (Plan B) ─────────────────────────
    # Bin edges are computed from the full trajectory X so that thresholds
    # reflect the true per-feature distribution, not just the 66 boundary
    # points.  bin_edges is saved as a pkl so the APG stage can reuse the
    # identical mapping when calling discretize(X_traj, bin_edges, ...).
    feature_names = list(FEATURE_COLS)
    labels        = BIN_LABELS[:n_bins]

    bin_edges = compute_bin_edges(X, feature_names, n_bins=n_bins)
    _save_pkl(
        bin_edges,
        os.path.join(output_dir, f"silver_{tag}_env_bin_edges.pkl"),
    )
    print(f"  Bin edges computed from {N} trajectory points "
          f"({n_bins} bins: {' / '.join(labels)})")

    # discretize boundary state — this is what the surrogate models receive
    bd_state_discrete = discretize(bd_state, bin_edges, feature_names)

    # save discretized boundary state for inspection
    disc_cols = [f"{f}_bin" for f in feature_names]
    bd_disc_df           = pd.DataFrame(bd_state_discrete, columns=disc_cols)
    bd_disc_df["ci"]     = ci_col
    bd_disc_df["cj"]     = cj_col
    bd_disc_df["action"] = bd_y
    _save_csv(
        bd_disc_df,
        os.path.join(output_dir, f"silver_{tag}_env_boundary_state_discrete.csv"),
    )
    print(f"  Discretized boundary state saved  "
          f"(shape={bd_state_discrete.shape})")

    # ── Step 6: surrogate models (trained on discretized boundary points) ─
    class_names = [str(a) for a in sorted(set(bd_y.tolist()))]

    # Decision Tree
    # Trained on bin indices; _save_tree_plot() converts thresholds to
    # real feature values (via bin_edges) purely for the rendered PDF.
    dt = _fit_decision_tree(bd_state_discrete, bd_y)
    _save_pkl(dt, os.path.join(output_dir, f"silver_{tag}_env_decision_tree.pkl"))
    _save_tree_plot(
        dt, feature_names, class_names,
        title=f"SILVER Decision Tree (env_action, real feature values) — {algo}",
        path=os.path.join(output_dir, f"silver_{tag}_env_decision_tree.pdf"),
        bin_edges=bin_edges,
    )

    # Linear Regression (trained on discrete bins)
    lr = _fit_linear_regression(bd_state_discrete, bd_y)
    _save_pkl(lr, os.path.join(output_dir, f"silver_{tag}_env_linear_regression.pkl"))

    # Sanity check: round+clip the continuous LR output on the training
    # (boundary) data back into a valid action index and compare against
    # the RL-guided labels bd_y.  This is the actual use of
    # _round_and_clip(): it turns the "continuous approximation" LR was
    # described as into a discrete action prediction, and we report how
    # often that recovers the true env_action label.
    lr_raw_preds    = lr.predict(bd_state_discrete)
    lr_action_preds = _round_and_clip(lr_raw_preds)
    lr_train_acc    = float(np.mean(lr_action_preds == bd_y))
    print(f"  Linear Regression (round+clip) training accuracy: "
          f"{lr_train_acc:.2%}  ({int(lr_train_acc * len(bd_y))}/{len(bd_y)})")

    # Logistic Regression (trained on discrete bins)
    if len(class_names) < 2:
        log_r = None
        print(f"  [WARNING] Only one action class ({class_names[0]}) among "
              f"boundary points — skipping Logistic Regression for {algo}.")
    else:
        log_r = _fit_logistic_regression(bd_state_discrete, bd_y)
        _save_pkl(log_r, os.path.join(output_dir, f"silver_{tag}_env_logistic_regression.pkl"))

    # ── Step 7: formula strings ───────────────────────────────────────────
    # Feature variable names for formula text; x1…x5 map to FEATURE_COLS.
    coeff_names = [f"x{i+1}" for i in range(len(feature_names))]

    formulas = []
    formulas.append(f"Feature mapping: {dict(enumerate(feature_names))}\n")

    # Discretization summary — maps bin index back to continuous interval,
    # e.g. feat_max_security_penalty: (0.34, 0.89] → medium (bin 1)
    formulas.append("=== Discretization Scheme ===")
    formulas.append(_bin_edges_summary(bin_edges, feature_names, n_bins, labels))

    formulas.append("\n=== Decision Tree (thresholds shown as real feature values) ===")
    dt_display = _tree_with_real_thresholds(dt, feature_names, bin_edges)
    formulas.append(
        export_text(dt_display, feature_names=feature_names,
                    decimals=TREE_DISPLAY_PRECISION)
    )

    formulas.append("\n=== Linear Regression (trained on discretized features) ===")
    formulas.append(_linear_formula(lr, coeff_names, fname="f"))
    formulas.append(
        f"Round+clip training accuracy on boundary points: {lr_train_acc:.2%} "
        f"({int(lr_train_acc * len(bd_y))}/{len(bd_y)} correct)"
    )

    formulas.append("\n=== Logistic Regression (trained on discretized features) ===")
    if log_r is None:
        formulas.append(
            f"Skipped: all {len(bd_y)} boundary points map to a single "
            f"action ({class_names[0]}); logistic regression requires "
            f"at least 2 classes."
        )
    else:
        formulas.append(_logistic_formulas(log_r, coeff_names, class_names, fname="f"))

    formula_path = os.path.join(output_dir, f"silver_{tag}_env_formulas.txt")
    with open(formula_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(formulas))
    print(f"  Formulas saved → {formula_path}")

    print(f"  ✓ SILVER (env_action, Plan-B discretization) complete for {algo}")
    return {
        "kmeans":             km,
        "bin_edges":          bin_edges,
        "dt":                 dt,
        "lr":                 lr,
        "log_r":              log_r,
        "lr_train_acc":       lr_train_acc,
        "bd_shap":            bd_shap,
        "bd_state":           bd_state,
        "bd_state_discrete":  bd_state_discrete,
        "bd_y":               bd_y,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALGOS = ["DQN", "Envelope", "EUPG", "PPO", "A2C"]


def run_all_env(
    df: pd.DataFrame,
    shap_dir: str,
    output_dir: str,
    n_clusters: int = None,
    n_bins: int = DEFAULT_N_BINS,
):
    os.makedirs(output_dir, exist_ok=True)
    for algo in df["algo"].unique():
        if algo not in ALGOS:
            print(f"[WARNING] Unknown algo '{algo}', skipping.")
            continue
        algo_df = df[df["algo"] == algo].reset_index(drop=True)
        run_silver_env(algo, algo_df, shap_dir, output_dir, n_clusters, n_bins)
    print(f"\n✓ All SILVER (env_action) analyses complete. Outputs in: {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SILVER with RL-Guided Labeling (env_action variant, Plan-B discretization)"
    )
    parser.add_argument("--input",      required=True,
                        help="Original data CSV (must contain env_action column)")
    parser.add_argument("--shap_dir",   required=True,
                        help="Directory containing shap_env_explain.py output CSVs "
                             "(flat structure, no subdirectories)")
    parser.add_argument("--output",     default="silver_env_outputs",
                        help="Output directory for SILVER env_action results")
    parser.add_argument("--algo",       default=None,
                        help="Run only one algo (DQN|Envelope|EUPG|PPO|A2C). "
                             "Omit to run all algos found in the CSV.")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help=f"KMeans k (default: N_ACTIONS={N_ACTIONS})")
    parser.add_argument("--n_bins",     type=int, default=DEFAULT_N_BINS,
                        help=f"Number of quantile bins per feature "
                             f"(default: {DEFAULT_N_BINS} → low / medium / high)")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    print(f"Loaded {len(data)} rows, algorithms: {data['algo'].unique().tolist()}")

    os.makedirs(args.output, exist_ok=True)

    if args.algo:
        algo_map = {a.lower(): a for a in ALGOS}
        args.algo = algo_map.get(args.algo.lower(), args.algo)
        algo_df = data[data["algo"] == args.algo].reset_index(drop=True)
        if algo_df.empty:
            raise ValueError(f"No rows found for algo='{args.algo}'")
        run_silver_env(
            args.algo, algo_df, args.shap_dir, args.output,
            args.n_clusters, args.n_bins,
        )
    else:
        run_all_env(data, args.shap_dir, args.output, args.n_clusters, args.n_bins)