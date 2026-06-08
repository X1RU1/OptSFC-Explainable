"""
APG Gen: Abstract Policy Graph Generation
Based on: Topin & Veloso, "Generation of Policy-Level Explanations for RL", AAAI 2019

Design B — behavior-centred, fully consistent across all algorithms:

  g(s)     = Q(s, a_executed)
             DQN / A2C / PPO  : best_q column directly
             Envelope / EUPG  : 0.4*best_weighted_q_resource
                               + 0.3*best_weighted_q_network
                               + 0.3*best_weighted_q_security
             (confirmed: best_weighted_q_* columns store the executed-action Q)

  grouping = env_action
             Abstract states are initialised by grouping transitions that share
             the same executed action.  Semantically consistent with g(s):
             "among states where action a was actually executed, which features
             most influence Q(s, a)?"

  edges    = empirical transition frequency
             P(b_j | b_i) = fraction of tuples in c_i whose s' maps to c_j.
             All transitions retained (off-policy transitions permitted per paper).

  epsilon  = 5th-percentile of |Q(s, a_executed) - Q(s, a_alt)|
             DQN / A2C / PPO  : quantile(|best_q - alt_q|, 0.05)
             Envelope / EUPG  : quantile(|g_value - alt_weighted_g|, 0.05)
             Same units as g(s) throughout.

  binarisation = per-feature median threshold
             Median preferred over mean: penalty/overhead features are
             right-skewed, and median keeps ~50 % of samples on each side,
             maximising the sqrt(p0*p1) term in FIRM.

  environment = continuous (non-episodic)
             All 5000 steps belong to a single episode; no environment resets.
             s' of step t is simply the feature vector of step t+1.
             Only the final row is dropped (no s' available).

Supports: dqn, envelope, a2c, ppo, eupg
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Objective-related features only (no resource-constraint features such as
# feat_vim0_cpu / feat_vim0_ram).
FEATURE_COLS = [
    "feat_mean_mtd_overhead",
    "feat_mean_network_penalty",
    "feat_max_network_penalty",
    "feat_mean_security_penalty",
    "feat_max_security_penalty",
]
BIN_COLS = [f + "_bin" for f in FEATURE_COLS]

N_ACTIONS = 12

# Multi-objective reward weights [resource, network, security] for
# Envelope and EUPG.  Must match the weights used during training.
REWARDS_COEFF = [0.4, 0.3, 0.3]

# Algorithms whose g(s) must be computed from per-objective Q columns.
MORL_ALGOS = {"envelope", "eupg"}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    s_bin: np.ndarray        # binarised state vector, shape (n_features,)
    action: int              # env_action — used for grouping and as g(s) anchor
    s_next_bin: np.ndarray   # binarised next-state vector (row t+1)
    g_value: float           # Q(s, a_executed)


@dataclass
class AbstractState:
    idx: int
    transitions: list = field(default_factory=list)
    # Feature-value constraints that define membership, e.g.
    # {"action": 3, "feat_mean_security_penalty": 1}
    # "action" key stores the initial env_action group.
    # Feature keys store the binary threshold side (0 = ≤ median, 1 = > median).
    constraints: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load data and compute g(s)
# ─────────────────────────────────────────────────────────────────────────────

def compute_g_value(df: pd.DataFrame, algo: str) -> pd.Series:
    """
    Compute g(s) = Q(s, a_executed) for every row.

    DQN / A2C / PPO:
        best_q already stores Q(s, env_action).

    Envelope / EUPG:
        best_weighted_q_{resource,network,security} store the per-objective
        Q values of the executed action.  We form the scalar weighted sum
        using REWARDS_COEFF.  This has been confirmed: these columns refer
        to the executed action, not the global argmax action.
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
        print(f"  g(s) = {REWARDS_COEFF[0]}*Q_resource"
              f" + {REWARDS_COEFF[1]}*Q_network"
              f" + {REWARDS_COEFF[2]}*Q_security  [executed action]")
    else:
        if "best_q" not in df.columns:
            raise ValueError(f"[{algo}] Column 'best_q' not found.")
        g = df["best_q"].copy()
        print(f"  g(s) = best_q  [Q of executed action]")
    return g


def load_data(csv_path: str, algo: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[{algo}] Loaded {len(df)} rows, {len(df.columns)} columns")

    missing_feat = [f for f in FEATURE_COLS if f not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing feature columns: {missing_feat}")

    df["g_value"] = compute_g_value(df, algo)

    # Informational: match rate between executed and reference action.
    # Not used for filtering — all transitions are retained.
    if "env_action" in df.columns and "reference_action" in df.columns:
        match = (df["env_action"] == df["reference_action"]).mean()
        print(f"  env_action == reference_action match rate: {match:.1%}")

    print(f"  g(s)  mean={df['g_value'].mean():.4f}"
          f"  std={df['g_value'].std():.4f}"
          f"  min={df['g_value'].min():.4f}"
          f"  max={df['g_value'].max():.4f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Binarise features
# ─────────────────────────────────────────────────────────────────────────────

def binarise_features(
    df: pd.DataFrame,
    thresholds: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Convert each continuous feature to a binary feature using its median as
    the threshold.  Median is preferred over mean because penalty and overhead
    features are right-skewed; using the median keeps approximately 50 % of
    samples on each side, which maximises the sqrt(p0*p1) term in FIRM and
    produces more stable importance estimates.

    If thresholds are provided (e.g. from a training split), they are applied
    directly without recomputing.
    """
    if thresholds is None:
        thresholds = {}
        print("  Feature binarisation thresholds (median):")
        for feat in FEATURE_COLS:
            t = float(df[feat].median())
            thresholds[feat] = t
            pct_high = (df[feat] > t).mean()
        print(f"    {feat:<40s}: {t:.4f}  ({pct_high:.1%} above threshold)"
              + ("  ⚠ near-constant, FIRM importance will be ~0"
                 if pct_high < 0.02 or pct_high > 0.98 else ""))

    for feat in FEATURE_COLS:
        df[feat + "_bin"] = (df[feat] > thresholds[feat]).astype(int)

    return df, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build transition tuples
# ─────────────────────────────────────────────────────────────────────────────

def build_transitions(df: pd.DataFrame) -> list[Transition]:
    """
    Construct (s_bin, env_action, s'_bin, g_value) tuples.

    The environment is continuous (non-episodic): all steps belong to a single
    episode with no resets.  s' for step t is the feature vector of step t+1.
    Only the final row is dropped because it has no successor.

    If a 'summary' column is present and marks episode boundaries, those rows
    are also dropped.  In the current setup this has no practical effect.

    Grouping key is env_action, consistent with g(s) = Q(s, a_executed).
    """
    df = df.sort_values("step").reset_index(drop=True)

    # The environment is continuous (non-episodic): all steps belong to a
    # single episode with no resets.  The 'summary' column contains per-step
    # explanation text and must NOT be used as an episode boundary marker.
    # Only the final row is dropped because it has no successor state s'.
    episode_end = pd.Series(False, index=df.index)
    episode_end.iloc[-1] = True
    print(f"  Rows dropped (last row only, no s' available): 1")

    transitions = []
    for i in range(len(df) - 1):
        if episode_end.iloc[i]:
            continue
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        transitions.append(Transition(
            s_bin=row[BIN_COLS].values.astype(int),
            action=int(row["env_action"]),
            s_next_bin=nxt[BIN_COLS].values.astype(int),
            g_value=float(row["g_value"]),
        ))

    print(f"  Transition tuples built: {len(transitions)}")
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — FIRM feature importance (Algorithm 3 in paper)
# ─────────────────────────────────────────────────────────────────────────────

def compute_firm(transitions: list[Transition]) -> np.ndarray:
    """
    Compute FIRM importance for every feature over a set of transitions.

        I_f(c) = |(q_f0 - q_f1) * sqrt(p_f0 * p_f1)|

    where
        p_fv = P(s[f] = v)
        q_fv = E[g(s) | s[f] = v]

    A feature that is constant within the set (p_f0 = 0 or p_f1 = 0) receives
    importance 0 and will never be selected for splitting.

    Returns an array of shape (n_features,).
    """
    n = len(transitions)
    if n == 0:
        return np.zeros(len(FEATURE_COLS))

    g_vals = np.array([t.g_value for t in transitions])   # (n,)
    states = np.array([t.s_bin for t in transitions])      # (n, F)

    importance = np.zeros(states.shape[1])
    for f in range(states.shape[1]):
        mask0 = states[:, f] == 0
        p0 = mask0.sum() / n
        p1 = 1.0 - p0
        if p0 == 0.0 or p1 == 0.0:
            continue
        q0 = g_vals[mask0].mean()
        q1 = g_vals[~mask0].mean()
        importance[f] = abs((q0 - q1) * np.sqrt(p0 * p1))

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Compute epsilon
# ─────────────────────────────────────────────────────────────────────────────

def compute_epsilon(df: pd.DataFrame, algo: str) -> float:
    """
    Compute the FIRM stopping threshold epsilon.

    Epsilon is the 5th percentile of |Q(s, a_executed) - Q(s, a_alt)| across
    all rows.  Using the 5th percentile rather than the minimum avoids extreme
    small values that would cause excessive splitting.

    Units are identical to g(s) (Q-value scale) for all algorithms, ensuring
    the stopping condition is meaningful and cross-algorithm comparable.

    DQN / A2C / PPO:
        delta = |best_q - alt_q|

    Envelope / EUPG:
        alt_g = 0.4*alt_weighted_q_resource
               + 0.3*alt_weighted_q_network
               + 0.3*alt_weighted_q_security
        delta = |g_value - alt_g|

    Fallback (if required columns are absent):
        epsilon = 5 % of g_value standard deviation.
    """
    if algo in MORL_ALGOS:
        needed = [
            "alt_weighted_q_resource",
            "alt_weighted_q_network",
            "alt_weighted_q_security",
        ]
        if all(c in df.columns for c in needed):
            alt_g = (
                REWARDS_COEFF[0] * df["alt_weighted_q_resource"] +
                REWARDS_COEFF[1] * df["alt_weighted_q_network"] +
                REWARDS_COEFF[2] * df["alt_weighted_q_security"]
            )
            delta = (df["g_value"] - alt_g).abs()
            delta = delta[delta > 1e-8]
            if len(delta) > 0:
                eps = float(delta.quantile(0.05))
                print(f"  epsilon = {eps:.6f}"
                      f"  [5th pct of |g_value - alt_weighted_g|]")
                return max(eps, 1e-6)
    else:
        if "best_q" in df.columns and "alt_q" in df.columns:
            delta = (df["best_q"] - df["alt_q"]).abs()
            delta = delta[delta > 1e-8]
            if len(delta) > 0:
                eps = float(delta.quantile(0.05))
                print(f"  epsilon = {eps:.6f}"
                      f"  [5th pct of |best_q - alt_q|]")
                return max(eps, 1e-6)

    # Fallback
    eps = max(float(df["g_value"].std() * 0.05), 1e-6)
    print(f"  epsilon = {eps:.6f}  [fallback: 5 % of g_value std]")
    return eps


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — APG Gen: divide abstract states (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def divide_abstract_states(
    transitions: list[Transition],
    epsilon: float,
    max_splits: int = 500,
) -> list[AbstractState]:
    """
    Algorithm 1: iterative greedy FIRM splitting.

    Initialisation:
        Transitions are grouped by env_action, producing up to N_ACTIONS
        initial abstract states.  This is consistent with g(s) = Q(s, a_executed):
        the initial partition asks "for each executed action, which states led
        to that action being taken?"

    Splitting:
        At each iteration, find the abstract state and feature with the highest
        FIRM importance.  Split that abstract state into two subsets along the
        binary feature boundary (value 0 vs value 1).  Recompute FIRM for the
        two new subsets.  Repeat until all importances are <= epsilon or
        max_splits is reached.

    Note: empty subsets after a split are discarded.  This can occur when all
    transitions in a set share the same binary value for the split feature.
    """
    # Initialise one abstract state per executed action.
    buckets: dict[int, list[Transition]] = {}
    for t in transitions:
        buckets.setdefault(t.action, []).append(t)

    idx = 0
    abstract_states: list[AbstractState] = []
    imp_cache: dict[int, np.ndarray] = {}

    for a, ts in sorted(buckets.items()):
        if ts:
            as_ = AbstractState(
                idx=idx,
                transitions=ts,
                constraints={"action": a},
            )
            abstract_states.append(as_)
            imp_cache[idx] = compute_firm(ts)
            idx += 1

    print(f"  Initial abstract states (one per env_action): {len(abstract_states)}")

    n_splits = 0
    while n_splits < max_splits:
        # Find the abstract state and feature with the highest importance.
        best_pos = -1
        best_imp = -1.0
        best_feat = -1

        for pos, as_ in enumerate(abstract_states):
            imp = imp_cache[as_.idx]
            fm = float(imp.max())
            if fm > best_imp:
                best_imp = fm
                best_feat = int(imp.argmax())
                best_pos = pos

        if best_imp <= epsilon:
            break  # All importances below threshold — stop.

        best_as = abstract_states[best_pos]
        feat_name = FEATURE_COLS[best_feat]
        base_constraints = dict(best_as.constraints)

        # Partition transitions on the binary value of best_feat.
        cn0_t = [t for t in best_as.transitions if t.s_bin[best_feat] == 0]
        cn1_t = [t for t in best_as.transitions if t.s_bin[best_feat] == 1]

        # Remove the parent abstract state.
        del imp_cache[best_as.idx]
        abstract_states.pop(best_pos)

        # Add non-empty children.
        for val, ts in [(0, cn0_t), (1, cn1_t)]:
            if ts:
                c = {**base_constraints, feat_name: val}
                as_new = AbstractState(idx=idx, transitions=ts, constraints=c)
                abstract_states.append(as_new)
                imp_cache[idx] = compute_firm(ts)
                idx += 1

        n_splits += 1

    print(f"  Splits performed: {n_splits}"
          f"  |  Final abstract states: {len(abstract_states)}")
    return abstract_states


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Build APG transition matrix (Algorithm 2)
# ─────────────────────────────────────────────────────────────────────────────

def build_apg(
    abstract_states: list[AbstractState],
) -> tuple[dict, np.ndarray]:
    """
    Algorithm 2: compute empirical transition frequency between abstract states.

    For each abstract state c_i:
        transition(i, j) = (number of tuples in c_i whose s' maps to c_j)
                           / |c_i|

    s' values not found in the lookup table (e.g. the very last observed state)
    are routed to the terminal abstract state b_T (index n).
    b_T has a self-loop with probability 1.

    Returns:
        lookup            : dict mapping tuple(s_bin) -> position in abstract_states
        transition_matrix : (n+1) x (n+1) numpy array
    """
    lookup: dict[tuple, int] = {}
    for pos, as_ in enumerate(abstract_states):
        for t in as_.transitions:
            lookup[tuple(t.s_bin)] = pos

    n = len(abstract_states)
    T = n  # index of terminal state b_T
    transition = np.zeros((n + 1, n + 1))

    for pos, as_ in enumerate(abstract_states):
        if not as_.transitions:
            continue
        for t in as_.transitions:
            next_pos = lookup.get(tuple(t.s_next_bin), T)
            transition[pos, next_pos] += 1.0 / len(as_.transitions)

    transition[T, T] = 1.0  # terminal self-loop (paper Algorithm 2, line 14)
    return lookup, transition


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Summarise and report
# ─────────────────────────────────────────────────────────────────────────────

def summarise(as_: AbstractState) -> dict:
    """Compute descriptive statistics for one abstract state."""
    if not as_.transitions:
        return {}
    actions = [t.action for t in as_.transitions]
    g_vals = [t.g_value for t in as_.transitions]
    ac = pd.Series(actions).value_counts()
    return {
        "n": len(as_.transitions),
        "dominant_action": int(ac.index[0]),
        "action_purity": float(ac.iloc[0] / len(actions)),
        "mean_g": float(np.mean(g_vals)),
        "std_g": float(np.std(g_vals)),
    }


def _shorten_feat(name: str) -> str:
    """Shorten a feature name for compact display."""
    return (name.replace("feat_", "")
                .replace("_penalty", "_pen")
                .replace("_overhead", "_oh")
                .replace("_network", "_net")
                .replace("_security", "_sec")
                .replace("mean_", "μ_")
                .replace("max_", "M_"))


def format_node_label(as_: AbstractState, thresholds: dict) -> str:
    """
    Format a concise multi-line label for a graph node.
    Constraints are shown with actual threshold values, e.g.
        μ_sec_pen ≤ 2.811
    instead of the raw binary indicator (0 or 1).
    """
    c = as_.constraints
    action = c.get("action", "?")
    feat_parts = []
    for k, v in c.items():
        if k == "action":
            continue
        op = ">" if v == 1 else "≤"
        thresh = thresholds.get(k, None)
        thresh_str = f"{thresh:.3f}" if thresh is not None else "θ"
        feat_parts.append(f"{_shorten_feat(k)}{op}{thresh_str}")
    s = summarise(as_)
    lines = [f"b{as_.idx}", f"a={action}", f"n={s.get('n', 0)}"] + feat_parts
    return "\n".join(lines)


def print_report(
    abstract_states: list[AbstractState],
    transition: np.ndarray,
    algo: str,
    thresholds: dict,
):
    n = len(abstract_states)
    print(f"\n{'='*68}")
    print(f"APG Report — {algo.upper()}")
    print(f"Abstract states (excluding terminal b_T): {n}")
    print(f"g(s) = Q(s, a_executed)   grouping = env_action   edges = frequency")
    print(f"Binarisation thresholds (median):")
    for feat, t in thresholds.items():
        print(f"  {feat:<40s}: {t:.4f}")
    print(f"{'='*68}")

    for pos, as_ in enumerate(abstract_states):
        s = summarise(as_)
        feat_c = {k: v for k, v in as_.constraints.items() if k != "action"}
        print(
            f"  b{as_.idx:03d}"
            f" | action_group={as_.constraints.get('action', '?')}"
            f" | dominant_a={s.get('dominant_action', '?')}"
            f"  purity={s.get('action_purity', 0):.0%}"
            f" | n={s.get('n', 0)}"
            f" | mean_g={s.get('mean_g', 0):.4f}"
            f"  std_g={s.get('std_g', 0):.4f}"
        )
        # Show human-readable constraints with actual threshold values.
        for feat, val in feat_c.items():
            op = ">" if val == 1 else "≤"
            thresh = thresholds.get(feat, None)
            thresh_str = f"{thresh:.4f}" if thresh is not None else "median"
            print(f"         {feat} {op} {thresh_str}")

        # Top outgoing transitions.
        row = transition[pos]
        tops = sorted(enumerate(row), key=lambda x: -x[1])
        tops = [(j, p) for j, p in tops if p >= 0.05][:4]
        for j, p in tops:
            dest = "b_T" if j == n else f"b{abstract_states[j].idx}"
            print(f"         → {dest}  p={p:.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Visualise
# ─────────────────────────────────────────────────────────────────────────────

def visualise_apg(
    abstract_states: list[AbstractState],
    transition: np.ndarray,
    algo: str,
    output_path: str,
    thresholds: dict,
    min_edge_prob: float = 0.05,
):
    """
    Draw the APG as a directed graph.
    Nodes are coloured by mean g(s) value (red = low, green = high).
    Edge width is proportional to transition probability.
    Edges with probability < min_edge_prob are hidden for clarity.
    Node labels show actual threshold values, e.g. mu_sec_pen <= 2.811.
    """
    n = len(abstract_states)
    G = nx.DiGraph()

    for pos, as_ in enumerate(abstract_states):
        s = summarise(as_)
        G.add_node(pos,
                   label=format_node_label(as_, thresholds),
                   mean_g=s.get("mean_g", 0.0))

    # Terminal node b_T.
    G.add_node(n, label="b_T\n(terminal)", mean_g=0.0)

    for i in range(n + 1):
        for j in range(n + 1):
            if i == n and j == n:
                continue  # skip terminal self-loop in visualisation
            p = transition[i, j]
            if p >= min_edge_prob:
                G.add_edge(i, j, weight=round(p, 3))

    fig_w = max(14, n * 1.2)
    fig_h = max(9, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    try:
        pos_layout = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos_layout = nx.spring_layout(G, seed=42, k=3.0)

    # Node colours: RdYlGn scaled by mean_g.
    g_vals = [G.nodes[i].get("mean_g", 0.0) for i in G.nodes()]
    g_min, g_max = min(g_vals), max(g_vals)
    g_range = g_max - g_min if g_max != g_min else 1.0
    colors = [
        plt.cm.RdYlGn((G.nodes[i].get("mean_g", 0.0) - g_min) / g_range)
        for i in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos_layout,
                           node_color=colors, node_size=2200,
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
    nx.draw_networkx_edge_labels(G, pos_layout,
                                 edge_labels=edge_labels,
                                 font_size=6, ax=ax)

    short_feats = [f.replace("feat_", "") for f in FEATURE_COLS]
    ax.set_title(
        f"Abstract Policy Graph — {algo.upper()}\n"
        f"{n} abstract states  |  "
        f"g(s) = Q(s, a_executed)  |  "
        f"grouping = env_action  |  "
        f"features: {short_feats}",
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

def run_apg_gen(
    csv_path: str,
    algo: str,
    output_dir: str = "apg_output",
    min_edge_prob: float = 0.05,
) -> dict:
    """
    Full APG Gen pipeline for one algorithm.
    Returns a dict with the APG components for downstream use.
    """
    print(f"\n{'─'*68}")
    print(f"APG Gen  —  {algo.upper()}  —  {csv_path}")
    print(f"{'─'*68}")

    df = load_data(csv_path, algo)

    print("Binarising features...")
    df, thresholds = binarise_features(df)

    print("Building transition tuples...")
    transitions = build_transitions(df)
    if not transitions:
        print("  No transitions built — skipping.")
        return {}

    epsilon = compute_epsilon(df, algo)

    print("Running FIRM splitting (Algorithm 1)...")
    abstract_states = divide_abstract_states(transitions, epsilon)

    print("Building transition matrix (Algorithm 2)...")
    lookup, transition_matrix = build_apg(abstract_states)

    print_report(abstract_states, transition_matrix, algo, thresholds)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_png = str(Path(output_dir) / f"apg_{algo}.png")
    print("Visualising...")
    visualise_apg(abstract_states, transition_matrix, algo,
                  out_png, thresholds, min_edge_prob)

    return {
        "algo": algo,
        "abstract_states": abstract_states,
        "transition_matrix": transition_matrix,
        "lookup": lookup,
        "thresholds": thresholds,
        "epsilon": epsilon,
        "n_states": len(abstract_states),
    }


def run_all(
    data_dir: str = ".",
    output_dir: str = "apg_output",
    algos: Optional[list] = None,
    min_edge_prob: float = 0.05,
) -> dict:
    """Run APG Gen for all specified algorithms and print a comparison summary."""
    if algos is None:
        algos = ["dqn", "envelope", "a2c", "ppo", "eupg"]

    results = {}
    for algo in algos:
        csv_path = Path(data_dir) / f"{algo}_explain.csv"
        if not csv_path.exists():
            print(f"[{algo}] File not found: {csv_path} — skipping.")
            continue
        try:
            r = run_apg_gen(str(csv_path), algo, output_dir, min_edge_prob)
            if r:
                results[algo] = r
        except Exception as e:
            import traceback
            print(f"[{algo}] ERROR: {e}")
            traceback.print_exc()

    if results:
        print(f"\n{'='*68}")
        print("Cross-algorithm comparison  (Design B — behavior-centred)")
        print(f"{'='*68}")
        print(f"{'Algo':<10} {'# States':>10} {'epsilon':>12}  g(s)")
        print("-" * 68)
        for algo, r in results.items():
            g_note = ("weighted Q of executed action"
                      if algo in MORL_ALGOS else
                      "best_q  (Q of executed action)")
            print(f"{algo:<10} {r['n_states']:>10}"
                  f" {r['epsilon']:>12.6f}  {g_note}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="APG Gen — Abstract Policy Graph generation for RL explanation"
    )
    parser.add_argument(
        "--data_dir", default=".",
        help="Directory containing {algo}_explain.csv files (default: .)"
    )
    parser.add_argument(
        "--output_dir", default="apg_output",
        help="Output directory for figures and reports (default: apg_output)"
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
        output_dir=args.output_dir,
        algos=args.algos,
        min_edge_prob=args.min_edge_prob,
    )