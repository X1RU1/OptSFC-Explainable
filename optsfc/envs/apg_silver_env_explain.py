"""
SILVER-driven Abstract Policy Graph (APG) Generation  —  env_action variant
=============================================================================
Design C (env_action) — SILVER decision-tree-driven abstraction.

This file is the env_action counterpart of apg_silver_explain.py (argmax
variant).  The only structural difference is that the SILVER decision trees
loaded here were trained on env_action labels (produced by
silver_env_explain.py), so silver_match_rate reflects how often the
policy's EXECUTED action agrees with SILVER's rule for that region — both
sides of the comparison are now grounded in the same env_action signal.

Difference from apg_silver_argmax_explain.py (argmax variant)
--------------------------------------------------------
  - Loads silver_<tag>_env_decision_tree.pkl  (from silver_env_outputs/)
    instead of silver_<tag>_decision_tree.pkl  (from silver_outputs/).
  - silver_match_rate semantics are stronger: SILVER was trained to predict
    env_action, so fidelity measures consistency between a model trained on
    executed actions and the executed actions themselves (expected to be
    higher than the argmax variant).
  - Default --silver_dir : silver_env_outputs
  - Default --output_dir : apg_silver_env_outputs
  - Output filenames     : silver_apg_<tag>_env.png
                           silver_apg_<tag>_env_assignments.csv

Discretization alignment (Plan B)
----------------------------------
silver_env_explain.py trains the decision tree on DISCRETIZED features
(integer bin indices: 0=low, 1=medium, 2=high per feature).  For
tree.apply() to assign leaves consistently, the same discretization must
be applied here before calling dt.apply().

The bin_edges dict (saved as silver_<tag>_env_bin_edges.pkl by
silver_env_explain.py) is loaded alongside the decision tree and passed to
the shared discretize() function so that leaf assignment in the APG stage
uses an identical feature representation to the surrogate's training.

Failure to discretize here would cause tree.apply() to use raw floating-
point feature values against split thresholds that were learned on integer
bin indices, producing semantically incorrect leaf assignments.

Rationale
---------
SILVER (env_action) already learned decision rules whose predicted Class
corresponds to the action the agent actually executed.  Using these leaves
as APG abstract states therefore gives a closed loop:

  env_action  →  shap_explain_env_action.py  →  Φ_s (sliced at env_action)
              →  silver_env_explain.py        →  decision tree (fit on env_action labels,
                                                  trained on discretized features)
              →  apg_silver_env_explain.py    →  APG (leaf = env_action decision region,
                                                  assigned via discretize → tree.apply())

silver_match_rate at each node measures within-region consistency of this
closed loop on the FULL trajectory (not just the boundary training points).

Inputs
------
  --data_dir   : directory containing {algo}_explain.csv
  --silver_dir : directory containing silver_<tag>_env_decision_tree.pkl
                 and silver_<tag>_env_bin_edges.pkl
                 (output of silver_env_explain.py, default: silver_env_outputs)

Outputs (per algorithm, written to --output_dir, default: apg_silver_env_outputs)
----------------------------------------------------------------------------------
  silver_apg_<tag>_env.png
  silver_apg_<tag>_env_assignments.csv

Supports: dqn, envelope, a2c, ppo, eupg
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — must match silver_env_explain.py / shap_explain_env_action.py
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "feat_mean_mtd_overhead",
    "feat_mean_network_penalty",
    "feat_max_network_penalty",
    "feat_mean_security_penalty",
    "feat_max_security_penalty",
]

N_ACTIONS = 12

# Multi-objective reward weights [resource, network, security]
REWARDS_COEFF = [0.4, 0.3, 0.3]

# Algorithms whose g(s) must be computed from per-objective Q columns.
MORL_ALGOS = {"envelope", "eupg"}

ALGOS = ["dqn", "envelope", "a2c", "ppo", "eupg"]


# ─────────────────────────────────────────────────────────────────────────────
# Discretization helper (mirrors silver_env_explain.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def discretize(X: np.ndarray, bin_edges: dict, feature_names: list) -> np.ndarray:
    """
    Map a continuous feature matrix to integer bin indices using the
    bin_edges dict saved by silver_env_explain.py.

    This function is copied verbatim from silver_env_explain.py and must
    remain identical to it.  Any change to the binning logic there must be
    reflected here, and vice versa.

    Bin assignment rule (matches np.digitize with right=True):
        bin 0  (low)    : v <= bin_edges[f][0]
        bin 1  (medium) : bin_edges[f][0] < v <= bin_edges[f][1]
        bin 2  (high)   : v > bin_edges[f][1]                     (for n_bins=3)

    Parameters
    ----------
    X            : (N, n_features) continuous feature matrix — raw trajectory
    bin_edges    : dict loaded from silver_<tag>_env_bin_edges.pkl
    feature_names: list of feature name strings matching X columns
                   (must equal FEATURE_COLS)

    Returns
    -------
    X_discrete : (N, n_features) integer array of bin indices (dtype int32)
    """
    X_discrete = np.zeros_like(X, dtype=np.int32)
    for col_idx, name in enumerate(feature_names):
        edges = bin_edges[name]
        X_discrete[:, col_idx] = np.digitize(X[:, col_idx], edges, right=True)
    return X_discrete


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load data and compute g(s)
# ─────────────────────────────────────────────────────────────────────────────

def compute_g_value(df: pd.DataFrame, algo: str) -> pd.Series:
    """
    g(s) = Q(s, a_executed).

    DQN / A2C / PPO  : best_q column directly.
    Envelope / EUPG  : weighted sum of best_weighted_q_{resource,network,security}.

    Used ONLY for node colouring and cross-method value comparison; plays no
    role in defining abstract states (those come from the SILVER tree).
    """
    if algo in MORL_ALGOS:
        needed = [
            "best_weighted_q_resource",
            "best_weighted_q_network",
            "best_weighted_q_security",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"[{algo}] Missing columns for g(s): {missing}")
        g = (
            REWARDS_COEFF[0] * df["best_weighted_q_resource"] +
            REWARDS_COEFF[1] * df["best_weighted_q_network"] +
            REWARDS_COEFF[2] * df["best_weighted_q_security"]
        )
    else:
        if "best_q" not in df.columns:
            raise ValueError(f"[{algo}] Column 'best_q' not found.")
        g = df["best_q"].copy()
    return g


def load_data(csv_path: str, algo: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[{algo}] Loaded {len(df)} rows, {len(df.columns)} columns")

    missing_feat = [f for f in FEATURE_COLS if f not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing feature columns: {missing_feat}")
    if "env_action" not in df.columns:
        raise ValueError("Missing 'env_action' column.")
    if "step" not in df.columns:
        raise ValueError("Missing 'step' column.")

    df["g_value"] = compute_g_value(df, algo)
    print(f"  g(s)  mean={df['g_value'].mean():.4f}"
          f"  std={df['g_value'].std():.4f}"
          f"  min={df['g_value'].min():.4f}"
          f"  max={df['g_value'].max():.4f}")

    df = df.sort_values("step").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load SILVER (env_action) decision tree, bin_edges, and leaf paths
# ─────────────────────────────────────────────────────────────────────────────

def load_decision_tree(tag: str, silver_dir: str):
    """
    Load silver_<tag>_env_decision_tree.pkl produced by silver_env_explain.py.

    The _env suffix distinguishes these trees (trained on env_action labels,
    using discretized features) from the argmax variant.
    """
    p = Path(silver_dir) / f"silver_{tag}_env_decision_tree.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"Decision tree not found: {p}\n"
            f"Expected silver_env_explain.py output at: {silver_dir}/"
        )
    with open(p, "rb") as f:
        dt = pickle.load(f)
    return dt


def load_bin_edges(tag: str, silver_dir: str) -> dict:
    """
    Load silver_<tag>_env_bin_edges.pkl produced by silver_env_explain.py.

    bin_edges maps each feature name to a 1-D array of (n_bins - 1) cut
    points, e.g. for n_bins=3:
        {"feat_mean_mtd_overhead": [0.23, 0.67], ...}

    These edges were computed from the full per-algo trajectory (not from
    the 66 boundary points), so they reflect the true feature distribution.
    They must be applied to the raw trajectory features here before calling
    dt.apply() so that the tree receives the same integer bin indices as
    during training.

    Raises FileNotFoundError if the pkl is missing, which means
    silver_env_explain.py has not yet been run for this algorithm.
    """
    p = Path(silver_dir) / f"silver_{tag}_env_bin_edges.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"Bin edges not found: {p}\n"
            f"Run silver_env_explain.py first to generate bin_edges for {tag}.\n"
            f"Expected file: {silver_dir}/silver_{tag}_env_bin_edges.pkl"
        )
    with open(p, "rb") as f:
        bin_edges = pickle.load(f)
    return bin_edges


def extract_leaf_paths(dt, feature_names: list) -> dict:
    """
    Recursively walk the tree and, for every leaf, return the ordered list
    of (feature_name, op, threshold) conditions along the root-to-leaf path.

    op is "<=" for the left branch and ">" for the right branch, matching
    sklearn's convention (and SILVER's export_text output).

    Because the tree was trained on discretized integer bin indices, the
    threshold values here are bin indices (e.g. 0.5 separates bin 0 from
    bin 1).  These raw (feature, op, threshold) tuples are translated into
    human-readable interval strings by translate_conditions() before display.
    """
    tree  = dt.tree_
    paths: dict[int, list] = {}

    def recurse(node: int, conditions: list):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1 and right == -1:
            paths[node] = conditions
            return
        feat   = feature_names[tree.feature[node]]
        thresh = float(tree.threshold[node])
        recurse(left,  conditions + [(feat, "<=", thresh)])
        recurse(right, conditions + [(feat, ">",  thresh)])

    recurse(0, [])
    return paths


# Bin index labels used by silver_env_explain.py (must stay in sync).
_BIN_LABELS = ["low", "medium", "high"]


def _bin_index(thresh: float, op: str) -> int:
    """
    Convert a sklearn threshold + branch operator to the bin index it selects.

    sklearn stores split thresholds as midpoints between consecutive integers
    (e.g. 0.5 between bin 0 and bin 1, 1.5 between bin 1 and bin 2).
    The left branch (<= thresh) selects all bins up to floor(thresh);
    the right branch (> thresh) selects all bins above floor(thresh).

    Returns the single bin index that the condition constrains to, i.e.:
        feat <= 0.5  →  bin 0  (the left branch covers only bin 0)
        feat >  0.5  →  bins 1, 2, …  — caller accumulates these per feature
    This function is only used by translate_conditions(), which folds multiple
    conditions on the same feature into one interval string.
    """
    return int(np.floor(thresh))


def translate_conditions(conditions: list, bin_edges: dict,
                         bin_labels: list = None) -> list:
    """
    Translate raw (feature_name, op, bin_index_threshold) path conditions
    produced by extract_leaf_paths() into human-readable interval strings.

    The decision tree was trained on integer bin indices (0, 1, 2, …), so
    its split thresholds are midpoints such as 0.5 or 1.5.  This function:
      1. Accumulates lower and upper bin bounds per feature across all
         conditions on that feature in the path.
      2. Looks up the corresponding continuous value intervals from bin_edges.
      3. Formats each feature as a single interval line, e.g.:
             M_sec_pen in (0.34, 0.89] medium

    Multiple sklearn conditions on the same feature (e.g. from nested splits)
    are merged into a single interval by taking the tightest lo/hi bounds.

    Parameters
    ----------
    conditions : list of (feature_name, op, threshold) from extract_leaf_paths()
    bin_edges  : dict loaded from silver_<tag>_env_bin_edges.pkl
                 maps feature_name → array of (n_bins - 1) cut points
    bin_labels : list of label strings, length = n_bins
                 (default: ["low", "medium", "high"])

    Returns
    -------
    lines : list of formatted condition strings, one per active feature,
            e.g. ["M_sec_pen in (0.34, 0.89] medium",
                  "μ_net_pen in (-∞, 0.12] low"]
    """
    if bin_labels is None:
        bin_labels = _BIN_LABELS

    # Determine n_bins from any feature's edge array length (+1).
    n_bins = len(next(iter(bin_edges.values()))) + 1

    # Per feature: track the tightest [lo_bin, hi_bin] range implied by all
    # conditions on that feature along this root-to-leaf path.
    # lo_bin is the lowest allowed bin (inclusive), hi_bin the highest.
    feat_bounds: dict[str, list] = {}   # feat → [lo_bin, hi_bin]

    for feat, op, thresh in conditions:
        if feat not in bin_edges:
            continue                    # skip features not in discretization
        split_bin = _bin_index(thresh, op)
        if feat not in feat_bounds:
            feat_bounds[feat] = [0, n_bins - 1]
        lo, hi = feat_bounds[feat]
        if op == "<=":
            # Left branch: bin must be <= split_bin
            hi = min(hi, split_bin)
        else:
            # Right branch: bin must be > split_bin
            lo = max(lo, split_bin + 1)
        feat_bounds[feat] = [lo, hi]

    lines = []
    for feat, (lo_bin, hi_bin) in feat_bounds.items():
        edges = bin_edges[feat]        # shape (n_bins - 1,)

        # Continuous lower bound: edge below lo_bin (or -∞ if lo_bin == 0)
        lo_val = edges[lo_bin - 1] if lo_bin > 0 else None
        # Continuous upper bound: edge at hi_bin (or +∞ if hi_bin == n_bins-1)
        hi_val = edges[hi_bin]     if hi_bin < n_bins - 1 else None

        lo_str = f"{lo_val:.2f}" if lo_val is not None else "-∞"
        hi_str = f"{hi_val:.2f}" if hi_val is not None else "+∞"

        # Interval notation: open on left (lo, hi], half-open on right (-∞, hi]
        interval = f"({lo_str}, {hi_str}]" if hi_val is not None \
                   else f"({lo_str}, +∞)"

        # Bin label: use lo_bin label when lo == hi (single bin), else range
        if lo_bin == hi_bin:
            label = bin_labels[lo_bin] if lo_bin < len(bin_labels) else str(lo_bin)
        else:
            lo_lbl = bin_labels[lo_bin] if lo_bin < len(bin_labels) else str(lo_bin)
            hi_lbl = bin_labels[hi_bin] if hi_bin < len(bin_labels) else str(hi_bin)
            label  = f"{lo_lbl}–{hi_lbl}"

        short = _shorten_feat(feat)
        lines.append(f"{short} in {interval} {label}")

    return lines


def leaf_predicted_class(dt, leaf_id: int) -> int:
    """
    SILVER's predicted class for a leaf = argmax of tree_.value at that leaf,
    mapped through dt.classes_.

    For the env_action variant, dt.classes_ contains the env_action values
    seen among the boundary training points, so this class is itself an
    env_action index.
    """
    value = dt.tree_.value[leaf_id][0]
    return int(dt.classes_[int(np.argmax(value))])


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Assign every row to a SILVER leaf (= abstract state)
# ─────────────────────────────────────────────────────────────────────────────

def assign_leaves(dt, df: pd.DataFrame, bin_edges: dict) -> np.ndarray:
    """
    Discretize the raw trajectory features and map every row to a tree leaf.

    The SILVER decision tree was trained on discretized integer bin indices
    (0=low, 1=medium, 2=high per feature), not on raw continuous values.
    Passing raw floats to tree.apply() would compare them against bin-index
    thresholds, producing incorrect leaf assignments.

    Correct procedure:
        1. Extract raw continuous features X from the trajectory DataFrame.
        2. Apply discretize(X, bin_edges, FEATURE_COLS) to obtain X_discrete
           (integer bin indices, identical encoding to the training data).
        3. Call dt.apply(X_discrete) to assign each row to a leaf.

    Parameters
    ----------
    dt        : DecisionTreeClassifier loaded from silver_<tag>_env_decision_tree.pkl
    df        : trajectory DataFrame containing FEATURE_COLS columns
    bin_edges : dict loaded from silver_<tag>_env_bin_edges.pkl
                (must be the same dict used during silver_env_explain.py training)

    Returns
    -------
    leaf_ids : (N,) int array of sklearn internal leaf node ids
    """
    X = df[FEATURE_COLS].values.astype(np.float64)
    X_discrete = discretize(X, bin_edges, FEATURE_COLS)
    return dt.apply(X_discrete)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build the transition matrix between SILVER-leaf abstract states
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(leaf_ids: np.ndarray, present_leaves: list):
    """
    Empirical transition frequency between consecutive time steps' leaves.

    Continuous, non-episodic environment: s' of step t is the feature vector
    of step t+1; only the final row has no s' and is dropped from the
    transition count.  dt.apply() always returns a valid leaf for every
    row, so no terminal sink state is needed.

    Returns
    -------
    transition : (n, n) row-stochastic matrix over present_leaves
    idx_map    : dict mapping raw leaf_id → abstract-state index (0..n-1)
    """
    idx_map = {leaf: i for i, leaf in enumerate(present_leaves)}
    n       = len(present_leaves)
    counts  = np.zeros((n, n))
    totals  = np.zeros(n)

    m = len(leaf_ids)
    for i in range(m - 1):
        a = idx_map[int(leaf_ids[i])]
        b = idx_map[int(leaf_ids[i + 1])]
        counts[a, b] += 1.0
        totals[a]    += 1.0

    transition = np.zeros((n, n))
    for i in range(n):
        if totals[i] > 0:
            transition[i] = counts[i] / totals[i]

    print(f"  Transition pairs built: {m - 1}  (last row dropped, no s')")
    return transition, idx_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Per-leaf statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarise_leaf(leaf_id: int, df: pd.DataFrame,
                   leaf_ids: np.ndarray, dt) -> dict:
    """
    Compute leaf statistics on the FULL trajectory.

    silver_class      : SILVER's predicted env_action for this leaf (fixed
                        by the trained tree, which was fit on env_action labels).
    dominant_action   : most frequently EXECUTED action (env_action) among
                        rows mapped to this leaf.
    dominant_purity   : fraction of rows whose env_action == dominant_action.
    silver_match_rate : fraction of rows whose env_action == silver_class.
                        Because the tree was trained to predict env_action,
                        this measures closed-loop consistency between the
                        SILVER rule and the executed trajectory — expected to
                        be higher than the argmax variant.
    mean_g / std_g    : g(s) = Q(s, a_executed), for node colouring and
                        cross-method value comparison.
    """
    mask         = leaf_ids == leaf_id
    n            = int(mask.sum())
    silver_class = leaf_predicted_class(dt, leaf_id)

    if n > 0:
        sub_actions       = df.loc[mask, "env_action"].astype(int)
        sub_g             = df.loc[mask, "g_value"]
        vc                = sub_actions.value_counts()
        dominant_action   = int(vc.index[0])
        dominant_purity   = float(vc.iloc[0] / n)
        silver_match_rate = float((sub_actions == silver_class).mean())
        mean_g            = float(sub_g.mean())
        std_g             = float(sub_g.std()) if n > 1 else 0.0
    else:
        dominant_action   = silver_class
        dominant_purity   = 0.0
        silver_match_rate = 0.0
        mean_g            = 0.0
        std_g             = 0.0

    return {
        "n":                 n,
        "silver_class":      silver_class,
        "dominant_action":   dominant_action,
        "dominant_purity":   dominant_purity,
        "silver_match_rate": silver_match_rate,
        "mean_g":            mean_g,
        "std_g":             std_g,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shorten_feat(name: str) -> str:
    return (name.replace("feat_", "")
                .replace("_penalty", "_pen")
                .replace("_overhead", "_oh")
                .replace("_network", "_net")
                .replace("_security", "_sec")
                .replace("mean_", "μ_")
                .replace("max_", "M_"))


def format_node_label(idx: int, leaf_id: int,
                      conditions: list, stats: dict,
                      bin_edges: dict) -> str:
    """
    Build the multi-line text label shown inside each APG node.

    Header lines:
        b<idx> (leaf <leaf_id>)
        SILVER a=<silver_class>
        n=<count>  match=<silver_match_rate>

    Condition lines (one per active feature, translated from bin indices to
    continuous intervals via translate_conditions):
        <short_feat> in (<lo>, <hi>] <label>
    e.g.:
        M_sec_pen in (0.34, 0.89] medium
        μ_net_pen in (-∞, 0.12] low

    Parameters
    ----------
    idx        : abstract-state index (0-based, display order)
    leaf_id    : sklearn internal leaf node id
    conditions : raw (feature, op, threshold) list from extract_leaf_paths()
    stats      : dict from summarise_leaf()
    bin_edges  : dict from silver_<tag>_env_bin_edges.pkl; used to map bin
                 index thresholds back to continuous value intervals
    """
    header = [
        f"b{idx} (leaf {leaf_id})",
        f"SILVER a={stats['silver_class']}",
        f"n={stats['n']}  match={stats['silver_match_rate']:.0%}",
    ]
    cond_lines = translate_conditions(conditions, bin_edges)
    return "\n".join(header + cond_lines)


def print_report(
    present_leaves: list,
    leaf_paths: dict,
    stats: dict,
    transition: np.ndarray,
    algo: str,
    n_leaves_total: int,
    bin_edges: dict,
):
    n = len(present_leaves)
    print(f"\n{'='*68}")
    print(f"SILVER-driven APG Report (env_action) — {algo.upper()}")
    print(f"SILVER tree leaves (total): {n_leaves_total}")
    print(f"Abstract states (leaves visited by this trajectory): {n}")
    print(f"g(s) = Q(s, a_executed)  |  edges = empirical transition frequency")
    print(f"silver_match_rate: env_action vs SILVER class (trained on env_action)")
    print(f"Conditions shown as continuous intervals (translated from bin indices).")
    print(f"{'='*68}")

    for idx, leaf_id in enumerate(present_leaves):
        s = stats[leaf_id]
        print(
            f"  b{idx:03d} (leaf {leaf_id:03d})"
            f" | SILVER a={s['silver_class']}"
            f" | dominant_a(executed)={s['dominant_action']}"
            f"  purity={s['dominant_purity']:.0%}"
            f" | silver_match={s['silver_match_rate']:.0%}"
            f" | n={s['n']}"
            f" | mean_g={s['mean_g']:.4f}  std_g={s['std_g']:.4f}"
        )
        # Translate bin-index conditions to continuous interval strings.
        cond_lines = translate_conditions(leaf_paths[leaf_id], bin_edges)
        if cond_lines:
            for line in cond_lines:
                print(f"         {line}")
        else:
            print(f"         (root leaf — no conditions)")

        row  = transition[idx]
        tops = sorted(enumerate(row), key=lambda x: -x[1])
        tops = [(j, p) for j, p in tops if p >= 0.05][:4]
        for j, p in tops:
            print(f"         → b{j:03d}  p={p:.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_silver_apg(
    present_leaves: list,
    leaf_paths: dict,
    stats: dict,
    transition: np.ndarray,
    algo: str,
    output_path: str,
    bin_edges: dict,
    min_edge_prob: float = 0.05,
):
    """
    Draw the SILVER-driven APG (env_action variant) as a directed graph.

    Node colour = mean g(s) (red = low, green = high), matching the colour
    convention of apg_silver_explain.py for visual comparability.

    Node labels show leaf conditions as human-readable continuous intervals
    (translated from bin indices via translate_conditions), e.g.:
        M_sec_pen in (0.34, 0.89] medium
        μ_net_pen in (-∞, 0.12] low
    This makes the APG directly interpretable without cross-referencing the
    formulas file.
    """
    n = len(present_leaves)
    G = nx.DiGraph()

    for idx, leaf_id in enumerate(present_leaves):
        G.add_node(idx,
                   label=format_node_label(
                       idx, leaf_id, leaf_paths[leaf_id], stats[leaf_id],
                       bin_edges),
                   mean_g=stats[leaf_id]["mean_g"])

    for i in range(n):
        for j in range(n):
            p = transition[i, j]
            if p >= min_edge_prob:
                G.add_edge(i, j, weight=round(float(p), 3))

    fig_w = max(14, n * 1.2)
    fig_h = max(9,  n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    try:
        pos_layout = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos_layout = nx.spring_layout(G, seed=42, k=3.0)

    g_vals  = [G.nodes[i]["mean_g"] for i in G.nodes()]
    g_min, g_max = min(g_vals), max(g_vals)
    g_range = g_max - g_min if g_max != g_min else 1.0
    colors  = [
        plt.cm.RdYlGn((G.nodes[i]["mean_g"] - g_min) / g_range)
        for i in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos_layout,
                           node_color=colors, node_size=2400,
                           alpha=0.88, ax=ax)
    nx.draw_networkx_labels(
        G, pos_layout,
        labels={i: G.nodes[i]["label"] for i in G.nodes()},
        font_size=6, ax=ax,
    )
    edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos_layout, ax=ax,
        arrowsize=14,
        width=[w * 3.5 for w in edge_weights],
        edge_color="steelblue", alpha=0.65,
        connectionstyle="arc3,rad=0.08",
    )
    edge_labels = {(u, v): f"{d['weight']:.2f}"
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos_layout, edge_labels=edge_labels,
                                 font_size=6, ax=ax)

    ax.set_title(
        f"SILVER-driven Abstract Policy Graph (env_action) — {algo.upper()}\n"
        f"{n} abstract states (SILVER env_action decision-tree leaves)  |  "
        f"node colour = mean g(s) = Q(s, a_executed)  |  "
        f"edges = empirical transition frequency  |  "
        f"node conditions shown as continuous value intervals",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_silver_apg_env(
    csv_path: str,
    algo: str,
    silver_dir: str = "silver_env_outputs",
    output_dir: str = "apg_silver_env_outputs",
    min_edge_prob: float = 0.05,
) -> dict:
    """Full SILVER-driven APG pipeline (env_action variant) for one algorithm."""
    tag = algo.lower()
    print(f"\n{'─'*68}")
    print(f"SILVER-APG (env_action)  —  {algo.upper()}  —  {csv_path}")
    print(f"{'─'*68}")

    df = load_data(csv_path, algo)

    print("Loading SILVER (env_action) decision tree and bin edges...")
    dt        = load_decision_tree(tag, silver_dir)
    bin_edges = load_bin_edges(tag, silver_dir)

    leaf_paths = extract_leaf_paths(dt, FEATURE_COLS)
    print(f"  SILVER decision tree loaded: {len(leaf_paths)} leaves (total)")
    print(f"  SILVER classes_: {dt.classes_.tolist()}")
    print(f"  Bin edges loaded for {len(bin_edges)} features "
          f"({len(next(iter(bin_edges.values()))) + 1} bins each)")

    print("Assigning rows to SILVER leaves "
          "(discretize features → tree.apply)...")
    leaf_ids       = assign_leaves(dt, df, bin_edges)
    present_leaves = sorted(set(int(x) for x in leaf_ids))
    print(f"  Leaves visited by this trajectory: "
          f"{len(present_leaves)} / {len(leaf_paths)}")

    print("Building transition matrix...")
    transition, idx_map = build_transition_matrix(leaf_ids, present_leaves)

    print("Computing per-leaf statistics...")
    stats = {
        leaf: summarise_leaf(leaf, df, leaf_ids, dt)
        for leaf in present_leaves
    }

    print_report(present_leaves, leaf_paths, stats,
                 transition, algo, len(leaf_paths), bin_edges)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = str(out_dir / f"silver_apg_{tag}_env.png")
    print("Visualising...")
    visualise_silver_apg(
        present_leaves, leaf_paths, stats, transition,
        algo, out_png, bin_edges, min_edge_prob,
    )

    out_csv = out_dir / f"silver_apg_{tag}_env_assignments.csv"
    assign_df = pd.DataFrame({
        "step":           df["step"],
        "leaf_id":        leaf_ids,
        "abstract_state": [idx_map[int(l)] for l in leaf_ids],
        "env_action":     df["env_action"],
        "silver_class":   [leaf_predicted_class(dt, int(l)) for l in leaf_ids],
        "g_value":        df["g_value"],
    })
    assign_df.to_csv(out_csv, index=False)
    print(f"  Leaf assignments saved → {out_csv}")

    return {
        "algo":              algo,
        "n_leaves_total":    len(leaf_paths),
        "n_leaves_present":  len(present_leaves),
        "present_leaves":    present_leaves,
        "leaf_paths":        leaf_paths,
        "stats":             stats,
        "transition":        transition,
        "idx_map":           idx_map,
    }


def run_all(
    data_dir: str = ".",
    silver_dir: str = "silver_env_outputs",
    output_dir: str = "apg_silver_env_outputs",
    algos: Optional[list] = None,
    min_edge_prob: float = 0.05,
) -> dict:
    """Run SILVER-driven APG (env_action) for all specified algorithms."""
    if algos is None:
        algos = ALGOS

    results = {}
    for algo in algos:
        csv_path = Path(data_dir) / f"{algo}_explain.csv"
        if not csv_path.exists():
            print(f"[{algo}] File not found: {csv_path} — skipping.")
            continue
        try:
            r = run_silver_apg_env(
                str(csv_path), algo, silver_dir, output_dir, min_edge_prob
            )
            if r:
                results[algo] = r
        except Exception as e:
            import traceback
            print(f"[{algo}] ERROR: {e}")
            traceback.print_exc()

    if results:
        print(f"\n{'='*68}")
        print("Cross-algorithm comparison  (Design C env_action — SILVER-leaf-driven)")
        print(f"{'='*68}")
        print(f"{'Algo':<10} {'tree leaves':>12} {'visited':>10} {'mean fidelity':>14}")
        print("-" * 68)
        for algo, r in results.items():
            match_rates   = [s["silver_match_rate"] for s in r["stats"].values()]
            mean_fidelity = float(np.mean(match_rates)) if match_rates else 0.0
            print(f"{algo:<10} {r['n_leaves_total']:>12} {r['n_leaves_present']:>10}"
                  f" {mean_fidelity:>13.1%}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SILVER-driven APG (env_action variant)"
    )
    parser.add_argument(
        "--data_dir", default=".",
        help="Directory containing {algo}_explain.csv files (default: .)"
    )
    parser.add_argument(
        "--silver_dir", default="silver_env_outputs",
        help="Directory containing silver_<tag>_env_decision_tree.pkl "
             "and silver_<tag>_env_bin_edges.pkl "
             "(default: silver_env_outputs)"
    )
    parser.add_argument(
        "--output_dir", default="apg_silver_env_outputs",
        help="Output directory for figures and assignment CSVs "
             "(default: apg_silver_env_outputs)"
    )
    parser.add_argument(
        "--algos", nargs="+", default=None,
        help="Algorithms to run, e.g. --algos dqn ppo  (default: all five)"
    )
    parser.add_argument(
        "--min_edge_prob", type=float, default=0.05,
        help="Minimum transition probability to draw as an edge (default: 0.05)"
    )
    args = parser.parse_args()

    run_all(
        data_dir=args.data_dir,
        silver_dir=args.silver_dir,
        output_dir=args.output_dir,
        algos=args.algos,
        min_edge_prob=args.min_edge_prob,
    )