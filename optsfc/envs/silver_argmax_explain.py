"""
SILVER with RL-Guided Labeling
================================
Builds a global interpretable surrogate policy from the Shapley vectors
(Φ_s) produced by shap_explain.py.

Pipeline (paper-aligned, DQN notebook)
---------------------------------------
Step 1  Load Φ_s and original state features X from shap_explain output
Step 2  KMeans on Φ_s  →  cluster labels + centroids
Step 3  Find boundary points in Φ_s space (one per cluster pair)
Step 4  Inverse mapping: boundary Φ_s  →  original state X
Step 5  RL-Guided Labeling: query action from original policy output
            DQN/Envelope → argmax Q
            PPO/A2C/EUPG → argmax prob
Step 6  Fit interpretable surrogate models on (boundary X, action labels):
            Decision Tree   (primary, fully interpretable)
            Linear Regression   (continuous approximation, round+clip)
            Logistic Regression (probabilistic multi-class)
Step 7  Save models, boundary datasets, and formula strings

Key design decisions vs paper
-------------------------------
- Paper uses a DNN to approximate the RL policy and query boundary labels.
  Here we skip the DNN and query action labels directly from the precomputed
  Q / probability columns in the original CSV  (exact policy, no approximation).
- Envelope: Φ_s comes from shap_envelope_scalar_Q.csv, computed by
  shap_explain.py on the scalarized Q vector (scalar_q_a{i} = w·Q_vec),
  with chosen action = argmax(scalar_q_a{i}) — structurally identical to
  DQN's q_ai_scalar. The action label here uses the SAME argmax
  (argmax(scalar_q_a{i})), so Φ_s and the action label always refer to the
  same chosen action. Envelope is therefore loaded through the same generic
  path as DQN/PPO/A2C/EUPG (see _load_shap), with no per-algo special case.
- All surrogate models are fit on original state features X (not Φ_s),
  matching the paper's shap_inverse_boundary_data path.

Input files (from shap_explain.py output dir)
----------------------------------------------
    shap_<tag>.csv          — Φ_s matrix, shape (N, n_features)
                              columns = FEATURE_COLS (5 objective features)
    For Envelope, <tag> = "envelope_scalar_Q" (i.e. shap_envelope_scalar_Q.csv).
    The diagnostic per-objective files (shap_envelope_Q_{resource,network,
    security}.csv) are NOT used here.

Input files (original data CSV, same as fed to shap_explain.py)
-----------------------------------------------------------------
    --input CSV             — must contain FEATURE_COLS + action Q/prob columns

Output files (per algo / tag)
------------------------------
    silver_<tag>_kmeans.pkl
    silver_<tag>_boundary_shap.csv      — boundary points in Φ_s space
    silver_<tag>_boundary_state.csv     — boundary points in state space (X)
    silver_<tag>_decision_tree.pkl
    silver_<tag>_linear_regression.pkl
    silver_<tag>_logistic_regression.pkl
    silver_<tag>_formulas.txt           — human-readable model equations

Usage
-----
    python silver_explain.py --input data.csv --shap_dir shap_outputs --output silver_outputs
    python silver_explain.py --input data.csv --shap_dir shap_outputs --output silver_outputs --algo DQN
"""

import os
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
# Constants — must match shap_explain.py
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "feat_mean_mtd_overhead",
    "feat_mean_network_penalty",
    "feat_max_network_penalty",
    "feat_mean_security_penalty",
    "feat_max_security_penalty",
]

N_ACTIONS          = 12
ACTION_COLS_PROB   = [f"prob_action_{i}" for i in range(N_ACTIONS)]
ACTION_COLS_SCALAR = [f"q_a{i}_scalar"   for i in range(N_ACTIONS)]
ENVELOPE_SCALAR    = [f"scalar_q_a{i}"   for i in range(N_ACTIONS)]

# SHAP output filename per algo tag — all algos now follow the same
# "one file = Φ_s for the policy's actually-selected action" convention.
# Envelope's scalar_q_a{i} is structurally identical to DQN's q_ai_scalar
# (both are all-action scalar value vectors with argmax = chosen action),
# so it is loaded through the exact same path (see _load_shap).
SHAP_FILE = {
    "DQN":      "shap_dqn_scalar_Q.csv",
    "PPO":      "shap_ppo_policy_prob.csv",
    "A2C":      "shap_a2c_policy_prob.csv",
    "EUPG":     "shap_eupg_policy_prob.csv",
    "Envelope": "shap_envelope_scalar_Q.csv",
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
    Load Φ_s matrix for algo. Files are under shap_dir/<algo_lower>/.

    Generic for all five algorithms (DQN, Envelope, PPO, A2C, EUPG):
    each reads a single SHAP_FILE[algo] whose Φ_s corresponds to the
    policy's actually-selected action (argmax Q / argmax prob), matching
    the action labels produced by _get_action_labels.
    """
    algo_dir = os.path.join(shap_dir, algo.lower())
    fname    = SHAP_FILE[algo]
    p        = os.path.join(algo_dir, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing SHAP file: {p}")
    return pd.read_csv(p)[FEATURE_COLS].values.astype(np.float64)


def _get_action_labels(algo: str, df: pd.DataFrame) -> np.ndarray:
    """
    Query action label for every state directly from precomputed Q/prob columns.
    Mirrors the paper's DNN query but uses exact policy outputs.
        DQN      → argmax scalar Q
        Envelope → argmax scalarized Q (scalar_q_a{i} = w·Q_vec), the same
                   argmax used to compute shap_envelope_scalar_Q.csv in
                   shap_explain.py — so Φ_s and the action label here always
                   refer to the same chosen action.
        PPO/A2C/EUPG → argmax softmax prob
    """
    if algo == "DQN":
        missing = [c for c in ACTION_COLS_SCALAR if c not in df.columns]
        if missing:
            raise ValueError(f"DQN Q columns missing: {missing}")
        return df[ACTION_COLS_SCALAR].values.argmax(axis=1)

    if algo == "Envelope":
        missing = [c for c in ENVELOPE_SCALAR if c not in df.columns]
        if missing:
            raise ValueError(f"Envelope scalar Q columns missing: {missing}")
        return df[ENVELOPE_SCALAR].values.argmax(axis=1)

    # PPO / A2C / EUPG
    missing = [c for c in ACTION_COLS_PROB if c not in df.columns]
    if missing:
        raise ValueError(f"{algo} prob columns missing: {missing}")
    return df[ACTION_COLS_PROB].values.argmax(axis=1)


# ---------------------------------------------------------------------------
# Step 2: KMeans on Φ_s
# ---------------------------------------------------------------------------

def _kmeans(shap_data: np.ndarray, n_clusters: int, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(shap_data)
    return km


# ---------------------------------------------------------------------------
# Step 3: Boundary point identification (paper: find_boundary_points)
# ---------------------------------------------------------------------------

def _find_boundary_points(shap_data: np.ndarray, centroids: np.ndarray):
    """
    For every pair of clusters (i, j), find the sample in Φ_s space
    whose distance to centroid_i and centroid_j is most equal
    (i.e. closest to the decision boundary between i and j).

    Returns list of (shap_vector, (cluster_i, cluster_j)).
    """
    n_clusters     = len(centroids)
    boundary_points = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dists     = cdist(shap_data, [centroids[i], centroids[j]], "euclidean")
            dist_diff = np.abs(dists[:, 0] - dists[:, 1])
            idx       = np.argmin(dist_diff)
            boundary_points.append((shap_data[idx], idx, (i, j)))

    return boundary_points   # list of (Φ_s vector, original_row_idx, (ci, cj))


# ---------------------------------------------------------------------------
# Step 4 + 5: Inverse mapping + RL-Guided Labeling
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
        - action label from policy   (RL-guided labeling, no DNN needed)

    Returns
    -------
    bd_shap  : (n_pairs, n_features) — boundary in Φ_s space
    bd_state : (n_pairs, n_features) — boundary in state space
    bd_y     : (n_pairs,)            — action labels
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
# Step 6: Surrogate models
# ---------------------------------------------------------------------------

def _fit_decision_tree(bd_X: np.ndarray, bd_y: np.ndarray) -> DecisionTreeClassifier:
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
    # coef_ shape: (1, n_features) for binary, (n_classes, n_features) for multiclass
    # Pad to (n_classes, n_features) so indexing is always consistent
    coef = log_r.coef_
    intercept = log_r.intercept_
    if coef.shape[0] == 1 and len(class_names) == 2:
        # binary: sklearn only stores one row; reconstruct both
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


def _save_csv(arr: np.ndarray, columns: list, path: str):
    pd.DataFrame(arr, columns=columns).to_csv(path, index=False)
    print(f"  Saved → {path}")


def _save_tree_plot(dt: DecisionTreeClassifier, feature_names: list,
                    class_names: list, title: str, path: str):
    fig, ax = plt.subplots(figsize=(max(20, len(class_names) * 4), 12))
    plot_tree(dt, feature_names=feature_names, class_names=class_names,
              filled=True, impurity=False, ax=ax)
    ax.set_title(title, fontsize=16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Per-algo SILVER runner
# ---------------------------------------------------------------------------

def run_silver(
    algo: str,
    df: pd.DataFrame,
    shap_dir: str,
    output_dir: str,
    n_clusters: int = None,
):
    """
    Full SILVER pipeline for one algorithm.

    Parameters
    ----------
    algo        : one of DQN / Envelope / PPO / A2C / EUPG
    df          : original data DataFrame (same CSV as shap_explain.py input)
    shap_dir    : directory containing shap_explain.py output CSVs
    output_dir  : where to write silver_* outputs
    n_clusters  : KMeans k  (default: N_ACTIONS = 12)
    """
    if n_clusters is None:
        n_clusters = N_ACTIONS

    tag = algo.lower() if algo != "Envelope" else "envelope"
    print(f"\n{'='*60}\nSILVER: {algo}  (k={n_clusters})\n{'='*60}")

    # ── Step 1: load Φ_s and state features ──────────────────────────────
    shap_data     = _load_shap(algo, shap_dir)          # (N, 5)
    X             = _load_state_features(df)             # (N, 5)
    action_labels = _get_action_labels(algo, df)         # (N,)
    N             = len(shap_data)
    print(f"  Loaded {N} samples,  Φ_s shape={shap_data.shape}")

    # ── Step 2: KMeans on Φ_s ────────────────────────────────────────────
    km        = _kmeans(shap_data, n_clusters)
    centroids = km.cluster_centers_
    _save_pkl(km, os.path.join(output_dir, f"silver_{tag}_kmeans.pkl"))
    print(f"  KMeans done — {n_clusters} clusters")

    # ── Step 3: boundary points ───────────────────────────────────────────
    boundary_points = _find_boundary_points(shap_data, centroids)
    n_pairs         = len(boundary_points)
    print(f"  Found {n_pairs} boundary points  (C({n_clusters},2)={n_pairs})")

    # ── Step 4+5: inverse mapping + RL-guided labeling ───────────────────
    bd_shap, bd_state, bd_y, meta = _build_boundary_dataset(
        boundary_points, X, shap_data, action_labels
    )

    # save boundary datasets
    shap_cols  = [f"shap_{f}" for f in FEATURE_COLS]
    state_cols = list(FEATURE_COLS)
    ci_col     = [m[0] for m in meta]
    cj_col     = [m[1] for m in meta]

    bd_shap_df           = pd.DataFrame(bd_shap,  columns=shap_cols)
    bd_shap_df["ci"]     = ci_col
    bd_shap_df["cj"]     = cj_col
    bd_shap_df["action"] = bd_y
    bd_shap_df.to_csv(
        os.path.join(output_dir, f"silver_{tag}_boundary_shap.csv"), index=False
    )

    bd_state_df           = pd.DataFrame(bd_state, columns=state_cols)
    bd_state_df["ci"]     = ci_col
    bd_state_df["cj"]     = cj_col
    bd_state_df["action"] = bd_y
    bd_state_df.to_csv(
        os.path.join(output_dir, f"silver_{tag}_boundary_state.csv"), index=False
    )
    print(f"  Boundary datasets saved  (n={len(bd_y)}, "
          f"unique actions={sorted(set(bd_y.tolist()))})")

    # ── Step 6: surrogate models (trained on state-space boundary points) ─
    feature_names = list(FEATURE_COLS)
    class_names   = [str(a) for a in sorted(set(bd_y.tolist()))]

    # Decision Tree
    dt = _fit_decision_tree(bd_state, bd_y)
    _save_pkl(dt, os.path.join(output_dir, f"silver_{tag}_decision_tree.pkl"))
    _save_tree_plot(
        dt, feature_names, class_names,
        title=f"SILVER Decision Tree — {algo}",
        path=os.path.join(output_dir, f"silver_{tag}_decision_tree.pdf"),
    )

    # Linear Regression
    lr = _fit_linear_regression(bd_state, bd_y)
    _save_pkl(lr, os.path.join(output_dir, f"silver_{tag}_linear_regression.pkl"))

    # Logistic Regression — sklearn's multinomial solver requires >= 2 classes.
    # For some algo/k combinations, every boundary point can collapse onto a
    # single action (the Φ_s-space cluster boundaries don't separate distinct
    # actions for this algo). In that case skip fitting and record why, rather
    # than crashing the whole run.
    if len(class_names) < 2:
        log_r = None
        print(f"  [WARNING] Only one action class ({class_names[0]}) among "
              f"boundary points — skipping Logistic Regression for {algo}.")
    else:
        log_r = _fit_logistic_regression(bd_state, bd_y)
        _save_pkl(log_r, os.path.join(output_dir, f"silver_{tag}_logistic_regression.pkl"))

    # ── Step 7: formula strings ───────────────────────────────────────────
    coeff_names = [f"x{i+1}" for i in range(len(feature_names))]
    formulas = []
    formulas.append(f"Feature mapping: {dict(enumerate(feature_names))}\n")
    formulas.append("=== Decision Tree ===")
    formulas.append(export_text(dt, feature_names=feature_names))
    formulas.append("\n=== Linear Regression ===")
    formulas.append(_linear_formula(lr, coeff_names, fname="f"))
    formulas.append("\n=== Logistic Regression ===")
    if log_r is None:
        formulas.append(
            f"Skipped: all {len(bd_y)} boundary points map to a single "
            f"action ({class_names[0]}); logistic regression requires "
            f"at least 2 classes."
        )
    else:
        formulas.append(_logistic_formulas(log_r, coeff_names, class_names, fname="f"))

    formula_path = os.path.join(output_dir, f"silver_{tag}_formulas.txt")
    with open(formula_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(formulas))
    print(f"  Formulas saved → {formula_path}")

    print(f"  ✓ SILVER complete for {algo}")
    return {
        "kmeans": km,
        "dt":     dt,
        "lr":     lr,
        "log_r":  log_r,
        "bd_shap":  bd_shap,
        "bd_state": bd_state,
        "bd_y":     bd_y,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALGOS = ["DQN", "Envelope", "EUPG", "PPO", "A2C"]


def run_all(df: pd.DataFrame, shap_dir: str, output_dir: str, n_clusters: int = None):
    os.makedirs(output_dir, exist_ok=True)
    for algo in df["algo"].unique():
        if algo not in ALGOS:
            print(f"[WARNING] Unknown algo '{algo}', skipping.")
            continue
        algo_df = df[df["algo"] == algo].reset_index(drop=True)
        run_silver(algo, algo_df, shap_dir, output_dir, n_clusters)
    print(f"\n✓ All SILVER analyses complete. Outputs in: {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SILVER with RL-Guided Labeling")
    parser.add_argument("--input",      required=True,
                        help="Original data CSV (same as fed to shap_explain.py)")
    parser.add_argument("--shap_dir",   required=True,
                        help="Directory containing shap_explain.py output CSVs")
    parser.add_argument("--output",     default="silver_outputs",
                        help="Output directory for SILVER results")
    parser.add_argument("--algo",       default=None,
                        help="Run only one algo (DQN|Envelope|EUPG|PPO|A2C). "
                             "Omit to run all algos found in the CSV.")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help=f"KMeans k (default: N_ACTIONS={N_ACTIONS})")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    print(f"Loaded {len(data)} rows, algorithms: {data['algo'].unique().tolist()}")

    os.makedirs(args.output, exist_ok=True)

    if args.algo:
        # normalise to match CSV casing: dqn→DQN, envelope→Envelope, etc.
        algo_map = {a.lower(): a for a in ALGOS}
        args.algo = algo_map.get(args.algo.lower(), args.algo)
        algo_df = data[data["algo"] == args.algo].reset_index(drop=True)
        if algo_df.empty:
            raise ValueError(f"No rows found for algo='{args.algo}'")
        run_silver(args.algo, algo_df, args.shap_dir, args.output, args.n_clusters)
    else:
        run_all(data, args.shap_dir, args.output, args.n_clusters)