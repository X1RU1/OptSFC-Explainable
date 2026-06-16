"""
SILVER-driven Abstract Policy Graph (APG) Generation
=====================================================
Design C — SILVER decision-tree-driven abstraction.

This is a SEPARATE pipeline from apg_explain.py (Design B, FIRM-driven). It does
NOT modify or overwrite the original APG results. It reads the SILVER
decision trees already produced by silver_explain.py and uses their leaf
structure directly as the abstract-state partition for the APG.

Rationale (per supervisor feedback)
------------------------------------
SILVER already learned a small set of human-readable decision rules
(root-to-leaf paths over FEATURE_COLS) whose predicted Class corresponds to
the policy's chosen action. Using these leaves as APG abstract states gives:

  - A much smaller, already-interpretable state count (bounded by the
    SILVER tree's leaf count, typically far below the ~100 states produced
    by FIRM splitting in Design B).
  - A direct SHAP -> SILVER -> APG closed loop: APG nodes ARE SILVER's
    decision regions, and APG edges show how the policy moves between these
    regions over time (a temporal/behavioural dimension SILVER alone does
    not provide).

What changes vs. Design B (apg_explain.py)
-----------------------------------------------------------------
  - NO FIRM splitting, NO epsilon, NO median binarisation, NO env_action
    based initial grouping. The partition is entirely inherited from the
    SILVER decision tree (`silver_<tag>_decision_tree.pkl`), applied on the
    RAW (continuous) FEATURE_COLS values via `tree.apply(X)`.
  - Each abstract state = one decision-tree leaf.
      * Node "rule"      = root-to-leaf path conditions (real thresholds,
                            same units as the SILVER tree was trained on).
      * Node "SILVER a=" = the leaf's predicted action (argmax of
                            tree_.value at that leaf) -- this is SILVER's
                            Class label, fixed by the trained tree.
      * Node "match="    = fraction of rows mapped to this leaf whose
                            ACTUAL executed action (env_action) equals the
                            SILVER class -- i.e. SILVER rule fidelity in
                            this region, recomputed on the full trajectory
                            (not just the 66 boundary points SILVER was
                            fit on).
  - g(s) = Q(s, a_executed) is still computed (same formula as Design B,
    reused from apg_explain.py) purely for node colouring / value-level
    comparison with the original APG and with RDX -- it plays NO role in
    the partitioning itself.
  - Edges = empirical transition frequency between consecutive time steps'
    leaves (continuous, non-episodic environment -- same convention as
    Design B: only the final row is dropped, no s' available for it).
    No terminal sink state b_T is needed here, because tree.apply() always
    returns a valid leaf for every row (no "off-grid" lookup misses).

Inputs
------
  --data_dir   : directory containing {algo}_explain.csv (same files used by
                 apg_explain.py / shap_explain.py / silver_explain.py)
  --silver_dir : directory containing silver_<tag>_decision_tree.pkl
                 (output of silver_explain.py)

Outputs (per algorithm, written to --output_dir, default "apg_silver_outputs")
------------------------------------------------------------------------------
  silver_apg_<tag>.png            -- graph visualisation
  silver_apg_<tag>_assignments.csv -- per-row leaf / abstract-state mapping
  (console report, can be redirected to a .txt if desired)

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
# Configuration -- must match apg_explain.py / shap_explain.py / silver_explain.py
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "feat_mean_mtd_overhead",
    "feat_mean_network_penalty",
    "feat_max_network_penalty",
    "feat_mean_security_penalty",
    "feat_max_security_penalty",
]

N_ACTIONS = 12

# Multi-objective reward weights [resource, network, security] for
# Envelope and EUPG -- must match training / apg_explain.py.
REWARDS_COEFF = [0.4, 0.3, 0.3]

# Algorithms whose g(s) must be computed from per-objective Q columns.
MORL_ALGOS = {"envelope", "eupg"}

# silver_explain.py tags (lowercase) -- identical to {algo}_explain.csv stems.
ALGOS = ["dqn", "envelope", "a2c", "ppo", "eupg"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load data and compute g(s)   (reused from apg_explain.py, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_g_value(df: pd.DataFrame, algo: str) -> pd.Series:
    """
    g(s) = Q(s, a_executed).

    DQN / A2C / PPO     : best_q column directly.
    Envelope / EUPG     : weighted sum of best_weighted_q_{resource,network,security}
                          using REWARDS_COEFF (executed-action Q, confirmed in
                          apg_explain.py).

    Used here ONLY for node colouring / cross-method value comparison -- it
    plays no role in defining the abstract states (those come from the
    SILVER tree).
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
# Step 2 — Load SILVER decision tree and extract leaf paths
# ─────────────────────────────────────────────────────────────────────────────

def load_decision_tree(tag: str, silver_dir: str):
    """Load silver_<tag>_decision_tree.pkl produced by silver_explain.py."""
    p = Path(silver_dir) / f"silver_{tag}_decision_tree.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Decision tree not found: {p}")
    with open(p, "rb") as f:
        dt = pickle.load(f)
    return dt


def extract_leaf_paths(dt, feature_names: list) -> dict:
    """
    Recursively walk the tree and, for every leaf, return the ordered list
    of (feature_name, op, threshold) conditions along the root-to-leaf path.

    op is "<=" for the left branch and ">" for the right branch, matching
    sklearn's convention (and the convention used in SILVER's
    export_text / formulas output).
    """
    tree = dt.tree_
    paths: dict[int, list] = {}

    def recurse(node: int, conditions: list):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1 and right == -1:
            paths[node] = conditions
            return
        feat = feature_names[tree.feature[node]]
        thresh = float(tree.threshold[node])
        recurse(left, conditions + [(feat, "<=", thresh)])
        recurse(right, conditions + [(feat, ">", thresh)])

    recurse(0, [])
    return paths


def leaf_predicted_class(dt, leaf_id: int) -> int:
    """
    SILVER's Class for a leaf = argmax of tree_.value at that leaf, mapped
    through dt.classes_ (which may be a strict subset of 0..N_ACTIONS-1,
    since it only contains actions seen among the 66 boundary points).
    """
    value = dt.tree_.value[leaf_id][0]
    return int(dt.classes_[int(np.argmax(value))])


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Assign every row to a SILVER leaf (= abstract state)
# ─────────────────────────────────────────────────────────────────────────────

def assign_leaves(dt, df: pd.DataFrame) -> np.ndarray:
    """
    Map every row's RAW (continuous) FEATURE_COLS values to a tree leaf id
    via tree.apply(). No binarisation -- the SILVER tree's own thresholds
    define the partition.
    """
    X = df[FEATURE_COLS].values.astype(np.float64)
    return dt.apply(X)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build the transition matrix between SILVER-leaf abstract states
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(leaf_ids: np.ndarray, present_leaves: list):
    """
    Empirical transition frequency between consecutive time steps' leaves.

    Continuous, non-episodic environment (same convention as apg_explain.py):
    s' of step t is the feature vector of step t+1; only the final row has
    no s' and is dropped from the transition count. Unlike Design B, no
    terminal sink b_T is needed: tree.apply() always returns a valid leaf
    for every row, so there is no "lookup miss" to route anywhere.

    Returns
    -------
    transition : (n, n) row-stochastic matrix over `present_leaves`
    idx_map    : dict mapping raw leaf_id -> abstract-state index (0..n-1)
    """
    idx_map = {leaf: i for i, leaf in enumerate(present_leaves)}
    n = len(present_leaves)
    counts = np.zeros((n, n))
    totals = np.zeros(n)

    m = len(leaf_ids)
    for i in range(m - 1):
        a = idx_map[int(leaf_ids[i])]
        b = idx_map[int(leaf_ids[i + 1])]
        counts[a, b] += 1.0
        totals[a] += 1.0

    transition = np.zeros((n, n))
    for i in range(n):
        if totals[i] > 0:
            transition[i] = counts[i] / totals[i]

    print(f"  Transition pairs built: {m - 1}  (last row dropped, no s')")
    return transition, idx_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Per-leaf statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarise_leaf(leaf_id: int, df: pd.DataFrame, leaf_ids: np.ndarray, dt) -> dict:
    """
    Recompute leaf statistics on the FULL trajectory (not just the 66
    boundary points SILVER was trained on).

    silver_class      : SILVER's predicted action for this leaf (fixed by
                         the trained tree).
    dominant_action   : most frequently EXECUTED action (env_action) among
                         rows mapped to this leaf.
    dominant_purity   : fraction of rows whose env_action == dominant_action.
    silver_match_rate : fraction of rows whose env_action == silver_class
                         -- i.e. how often the policy actually did what
                         SILVER's rule for this region predicts.
    mean_g / std_g    : g(s) = Q(s, a_executed), for colouring / value
                         comparison with the original APG and RDX.
    """
    mask = leaf_ids == leaf_id
    n = int(mask.sum())
    silver_class = leaf_predicted_class(dt, leaf_id)

    if n > 0:
        sub_actions = df.loc[mask, "env_action"].astype(int)
        sub_g = df.loc[mask, "g_value"]
        vc = sub_actions.value_counts()
        dominant_action = int(vc.index[0])
        dominant_purity = float(vc.iloc[0] / n)
        silver_match_rate = float((sub_actions == silver_class).mean())
        mean_g = float(sub_g.mean())
        std_g = float(sub_g.std()) if n > 1 else 0.0
    else:
        dominant_action = silver_class
        dominant_purity = 0.0
        silver_match_rate = 0.0
        mean_g = 0.0
        std_g = 0.0

    return {
        "n": n,
        "silver_class": silver_class,
        "dominant_action": dominant_action,
        "dominant_purity": dominant_purity,
        "silver_match_rate": silver_match_rate,
        "mean_g": mean_g,
        "std_g": std_g,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shorten_feat(name: str) -> str:
    """Shorten a feature name for compact display (same convention as apg_explain.py)."""
    return (name.replace("feat_", "")
                .replace("_penalty", "_pen")
                .replace("_overhead", "_oh")
                .replace("_network", "_net")
                .replace("_security", "_sec")
                .replace("mean_", "μ_")
                .replace("max_", "M_"))


def format_node_label(idx: int, leaf_id: int, conditions: list, stats: dict) -> str:
    lines = [
        f"b{idx} (leaf {leaf_id})",
        f"SILVER a={stats['silver_class']}",
        f"n={stats['n']}  match={stats['silver_match_rate']:.0%}",
    ]
    for feat, op, thresh in conditions:
        lines.append(f"{_shorten_feat(feat)}{op}{thresh:.3f}")
    return "\n".join(lines)


def print_report(
    present_leaves: list,
    leaf_paths: dict,
    stats: dict,
    transition: np.ndarray,
    algo: str,
    n_leaves_total: int,
):
    n = len(present_leaves)
    print(f"\n{'='*68}")
    print(f"SILVER-driven APG Report — {algo.upper()}")
    print(f"SILVER tree leaves (total): {n_leaves_total}")
    print(f"Abstract states (leaves visited by this trajectory): {n}")
    print(f"g(s) = Q(s, a_executed)  |  edges = empirical transition frequency")
    print(f"{'='*68}")

    for idx, leaf_id in enumerate(present_leaves):
        s = stats[leaf_id]
        cond_str = ", ".join(
            f"{feat} {op} {thresh:.4f}" for feat, op, thresh in leaf_paths[leaf_id]
        )
        print(
            f"  b{idx:03d} (leaf {leaf_id:03d})"
            f" | SILVER a={s['silver_class']}"
            f" | dominant_a(executed)={s['dominant_action']}"
            f"  purity={s['dominant_purity']:.0%}"
            f" | silver_match={s['silver_match_rate']:.0%}"
            f" | n={s['n']}"
            f" | mean_g={s['mean_g']:.4f}  std_g={s['std_g']:.4f}"
        )
        print(f"         rule: {cond_str if cond_str else '(root leaf — no conditions)'}")

        row = transition[idx]
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
    min_edge_prob: float = 0.05,
):
    """
    Draw the SILVER-driven APG as a directed graph.

    Nodes are coloured by mean g(s) (red = low, green = high), matching the
    colour convention of apg_explain.py's visualise_apg for visual comparability.
    Node labels show SILVER's predicted action, fidelity (silver_match), and
    the real-valued root-to-leaf rule conditions inherited from the tree.
    """
    n = len(present_leaves)
    G = nx.DiGraph()

    for idx, leaf_id in enumerate(present_leaves):
        G.add_node(idx,
                    label=format_node_label(idx, leaf_id, leaf_paths[leaf_id], stats[leaf_id]),
                    mean_g=stats[leaf_id]["mean_g"])

    for i in range(n):
        for j in range(n):
            p = transition[i, j]
            if p >= min_edge_prob:
                G.add_edge(i, j, weight=round(float(p), 3))

    fig_w = max(14, n * 1.2)
    fig_h = max(9, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    try:
        pos_layout = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos_layout = nx.spring_layout(G, seed=42, k=3.0)

    g_vals = [G.nodes[i]["mean_g"] for i in G.nodes()]
    g_min, g_max = min(g_vals), max(g_vals)
    g_range = g_max - g_min if g_max != g_min else 1.0
    colors = [
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
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos_layout, edge_labels=edge_labels,
                                   font_size=6, ax=ax)

    ax.set_title(
        f"SILVER-driven Abstract Policy Graph — {algo.upper()}\n"
        f"{n} abstract states (SILVER decision-tree leaves)  |  "
        f"node colour = mean g(s) = Q(s, a_executed)  |  "
        f"edges = empirical transition frequency",
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

def run_silver_apg(
    csv_path: str,
    algo: str,
    silver_dir: str = "silver_outputs",
    output_dir: str = "apg_silver_outputs",
    min_edge_prob: float = 0.05,
) -> dict:
    """Full SILVER-driven APG pipeline for one algorithm."""
    tag = algo.lower()
    print(f"\n{'─'*68}")
    print(f"SILVER-APG  —  {algo.upper()}  —  {csv_path}")
    print(f"{'─'*68}")

    df = load_data(csv_path, algo)

    print("Loading SILVER decision tree...")
    dt = load_decision_tree(tag, silver_dir)
    leaf_paths = extract_leaf_paths(dt, FEATURE_COLS)
    print(f"  SILVER decision tree loaded: {len(leaf_paths)} leaves (total)")
    print(f"  SILVER classes_: {dt.classes_.tolist()}")

    print("Assigning rows to SILVER leaves (tree.apply on raw features)...")
    leaf_ids = assign_leaves(dt, df)
    present_leaves = sorted(set(int(x) for x in leaf_ids))
    print(f"  Leaves visited by this trajectory: {len(present_leaves)} / {len(leaf_paths)}")

    print("Building transition matrix...")
    transition, idx_map = build_transition_matrix(leaf_ids, present_leaves)

    print("Computing per-leaf statistics...")
    stats = {leaf: summarise_leaf(leaf, df, leaf_ids, dt) for leaf in present_leaves}

    print_report(present_leaves, leaf_paths, stats, transition, algo, len(leaf_paths))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = str(out_dir / f"silver_apg_{tag}.png")
    print("Visualising...")
    visualise_silver_apg(present_leaves, leaf_paths, stats, transition, algo, out_png, min_edge_prob)

    out_csv = out_dir / f"silver_apg_{tag}_assignments.csv"
    assign_df = pd.DataFrame({
        "step": df["step"],
        "leaf_id": leaf_ids,
        "abstract_state": [idx_map[int(l)] for l in leaf_ids],
        "env_action": df["env_action"],
        "silver_class": [leaf_predicted_class(dt, int(l)) for l in leaf_ids],
        "g_value": df["g_value"],
    })
    assign_df.to_csv(out_csv, index=False)
    print(f"  Leaf assignments saved → {out_csv}")

    return {
        "algo": algo,
        "n_leaves_total": len(leaf_paths),
        "n_leaves_present": len(present_leaves),
        "present_leaves": present_leaves,
        "leaf_paths": leaf_paths,
        "stats": stats,
        "transition": transition,
        "idx_map": idx_map,
    }


def run_all(
    data_dir: str = ".",
    silver_dir: str = "silver_outputs",
    output_dir: str = "apg_silver_outputs",
    algos: Optional[list] = None,
    min_edge_prob: float = 0.05,
) -> dict:
    """Run SILVER-driven APG for all specified algorithms and print a comparison summary."""
    if algos is None:
        algos = ALGOS

    results = {}
    for algo in algos:
        csv_path = Path(data_dir) / f"{algo}_explain.csv"
        if not csv_path.exists():
            print(f"[{algo}] File not found: {csv_path} — skipping.")
            continue
        try:
            r = run_silver_apg(str(csv_path), algo, silver_dir, output_dir, min_edge_prob)
            if r:
                results[algo] = r
        except Exception as e:
            import traceback
            print(f"[{algo}] ERROR: {e}")
            traceback.print_exc()

    if results:
        print(f"\n{'='*68}")
        print("Cross-algorithm comparison  (Design C — SILVER-leaf-driven)")
        print(f"{'='*68}")
        print(f"{'Algo':<10} {'tree leaves':>12} {'visited':>10} {'mean fidelity':>14}")
        print("-" * 68)
        for algo, r in results.items():
            match_rates = [s["silver_match_rate"] for s in r["stats"].values()]
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
        description="SILVER-driven Abstract Policy Graph (APG) generation"
    )
    parser.add_argument(
        "--data_dir", default=".",
        help="Directory containing {algo}_explain.csv files (default: .)"
    )
    parser.add_argument(
        "--silver_dir", default="silver_outputs",
        help="Directory containing silver_<tag>_decision_tree.pkl (default: silver_outputs)"
    )
    parser.add_argument(
        "--output_dir", default="apg_silver_outputs",
        help="Output directory for figures and assignment CSVs (default: apg_silver_outputs)"
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