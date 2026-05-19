"""
Single-Step RDX Analysis
────────────────────────
Comprehensive single-step explanation for both single-objective RL and
multi-objective RL (MORL) algorithms, including:

  Value-based (MORL)        : Envelope  (per-objective Q)
  Value-based (single-obj)  : DQN       (scalar Q)
  Policy-based (MORL)       : EUPG
  Policy-based (single-obj) : PPO, A2C

Plot routing per algorithm
──────────────────────────
  Envelope / EUPG  →  q_landscape, q_diff, state_context, pairwise_rdx,
                       feat_action_corr
                       + feat_q_corr    (Envelope only)
                       + feat_prob_corr (EUPG only)

  DQN              →  state_context, feat_action_corr, feat_q_corr

  PPO / A2C        →  state_context, feat_action_corr, feat_prob_corr

Correlation plot routing
────────────────────────
  plot_feat_action_corr : ALL algorithms
      Feature ↔ binary indicator (env_action == i).
      Shows which state features drive action selection regardless of
      whether the algorithm is value-based or policy-based.

  plot_feat_q_corr      : VALUE-BASED only  (DQN, Envelope)
      Feature ↔ per-objective (or scalar) Q-value columns.
      Only meaningful when explicit Q estimates are logged.

  plot_feat_prob_corr   : POLICY-BASED only  (PPO, A2C, EUPG)
      Feature ↔ policy action-probability columns.
      Only meaningful when the actor outputs an explicit probability
      distribution over actions.

Usage
─────
  python morl_single_step.py \
      --envelope envelope_explain.csv \
      --eupg     eupg_explain.csv    \
      --dqn      dqn_explain.csv     \
      --ppo      ppo_explain.csv     \
      --a2c      a2c_explain.csv     \
      --step     42                  \   # omit for auto-selection
      --out      ./single_step_plots

Step auto-selection (match=True rows only):
  The step whose total absolute weighted Q-diff magnitude is largest,
  i.e. argmax(|Δ_resource| + |Δ_network| + |Δ_security|).
  For algorithms without diff columns, the first match=1 row is used.

Outputs (algorithm-dependent)
──────────────────────────────
  {out}/{algo}_step{N}_q_landscape.png     <- Envelope, EUPG only
  {out}/{algo}_step{N}_q_diff.png          <- Envelope, EUPG only
  {out}/{algo}_step{N}_pairwise_rdx.png    <- Envelope, EUPG only
  {out}/{algo}_step{N}_state_context.png   <- all algorithms
  {out}/{algo}_step{N}_feat_action_corr.png<- all algorithms
  {out}/{algo}_step{N}_feat_q_corr.png     <- value-based only (DQN, Envelope)
  {out}/{algo}_step{N}_feat_prob_corr.png  <- policy-based only (PPO, A2C, EUPG)
  {out}/regime_summary.csv
"""

import argparse
import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS    = np.array([0.4, 0.3, 0.3])
OBJ_NAMES  = ["Resource", "Network", "Security"]
OBJ_COLORS = {"Resource": "#2196F3", "Network": "#FF9800", "Security": "#4CAF50"}

# Algorithms that maintain explicit per-action Q-value estimates.
# Receive: state_context, feat_action_corr, feat_q_corr.
# Envelope additionally receives: q_landscape, q_diff, pairwise_rdx.
VALUE_BASED_ALGOS = {"DQN", "Envelope"}

# Algorithms that output an explicit probability distribution over actions.
# Receive: state_context, feat_action_corr, feat_prob_corr.
# EUPG additionally receives: q_landscape, q_diff, pairwise_rdx.
POLICY_BASED_ALGOS = {"PPO", "A2C", "EUPG"}

# Algorithms that produce the full per-step Q-landscape / RDX plots.
# DQN and PPO/A2C do not log per-action Q matrices in a multi-objective sense,
# so those heavy plots are skipped for them.
MORL_ALGOS = {"Envelope", "EUPG"}

# 22 state features grouped by semantic category
FEAT_META = [
    # (column_name, display_label, category)
    ("feat_vim0_cpu",               "VIM0 CPU",                "Resource Load"),
    ("feat_vim0_ram",               "VIM0 RAM (GB)",           "Resource Load"),
    ("feat_vim1_cpu",               "VIM1 CPU",                "Resource Load"),
    ("feat_vim1_ram",               "VIM1 RAM (GB)",           "Resource Load"),
    ("feat_max_apt_score",          "Max APT Score",           "Security"),
    ("feat_mean_apt_score",         "Mean APT Score",          "Security"),
    ("feat_max_dataleak_score",     "Max Data-Leak Score",     "Security"),
    ("feat_mean_dataleak_score",    "Mean Data-Leak Score",    "Security"),
    ("feat_max_dos_score",          "Max DoS Score",           "Security"),
    ("feat_mean_dos_score",         "Mean DoS Score",          "Security"),
    ("feat_mean_security_penalty",  "Mean Sec. Penalty",       "Security"),
    ("feat_max_security_penalty",   "Max Sec. Penalty",        "Security"),
    ("feat_security_penalty_cumul", "Cumul. Sec. Penalty",     "Security"),
    ("feat_mean_network_penalty",   "Mean Net. Penalty",       "Network"),
    ("feat_max_network_penalty",    "Max Net. Penalty",        "Network"),
    ("feat_mean_mtd_overhead",      "Mean MTD Overhead",       "MTD Budget"),
    ("feat_min_remaining_mig",      "Min Remaining Mig.",      "MTD Budget"),
    ("feat_mean_remaining_mig",     "Mean Remaining Mig.",     "MTD Budget"),
    ("feat_min_remaining_reinst",   "Min Remaining Reinst.",   "MTD Budget"),
    ("feat_mean_remaining_reinst",  "Mean Remaining Reinst.",  "MTD Budget"),
    # ("feat_steps_since_last_mtd",   "Steps Since Last MTD",    "Temporal"),
    ("feat_total_ues",              "Total UEs",               "Temporal"),
    ("feat_nb_resources",           "Nb. Resources",           "Temporal"),
]

CATEGORY_COLORS = {
    "Resource Load": "#2196F3",
    "Security":      "#F44336",
    "Network":       "#FF9800",
    "MTD Budget":    "#9C27B0",
    "Temporal":      "#607D8B",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame | None:
    if path is None or not os.path.isfile(path):
        print(f"  [skip] {label}: file not found ({path})")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"  [ok]   {label}: {len(df):,} rows  <-  {path}")
    return df


def n_actions(df: pd.DataFrame) -> int:
    """
    Infer the number of actions from logged columns, trying three formats
    in order of precedence:
      1. q_a{i}_resource  -- Envelope (MORL, per-objective Q)
      2. q_a{i}_scalar    -- DQN (single-objective Q)
      3. prob_action_{i}  -- PPO / A2C / EUPG (policy probability)
    """
    for suffix in ("_resource", "_scalar"):
        cols = [c for c in df.columns
                if c.startswith("q_a") and c.endswith(suffix)]
        if cols:
            return len(cols)
    prob_cols = [c for c in df.columns if c.startswith("prob_action_")]
    return len(prob_cols)


def get_q_matrix(row: pd.Series, n_act: int) -> np.ndarray:
    """
    Extract an (n_actions, 3) Q matrix from a single log row.

    Column format handling:
      - q_a{i}_resource / _network / _security  ->  Envelope (MORL)
      - q_a{i}_scalar                           ->  DQN; value broadcast to
                                                     all three objective slots
      - neither present                          ->  PPO / A2C (no per-action Q);
                                                     returns zero matrix

    Returns shape (n_act, 3) as float64.
    """
    mat = np.zeros((n_act, 3), dtype=np.float64)
    for i in range(n_act):
        if f"q_a{i}_scalar" in row.index:
            v = float(row[f"q_a{i}_scalar"])
            mat[i] = [v, v, v]          # single-objective: broadcast to all 3
        elif f"q_a{i}_resource" in row.index:
            mat[i, 0] = float(row[f"q_a{i}_resource"])
            mat[i, 1] = float(row[f"q_a{i}_network"])
            mat[i, 2] = float(row[f"q_a{i}_security"])
        # else: policy-based with no per-action Q -- leave as zeros
    return mat


def scalar_q(q_mat: np.ndarray) -> np.ndarray:
    """Compute scalarised Q = Q @ WEIGHTS for each action. Shape (n_act,)."""
    return q_mat @ WEIGHTS


def get_probs(row: pd.Series, n_act: int) -> np.ndarray | None:
    """
    Extract policy action probabilities from a log row.
    Returns a float array of shape (n_act,), or None if the columns are absent.
    Present for policy-based algorithms (PPO, A2C, EUPG); absent for
    value-based algorithms (DQN, Envelope).
    """
    cols = [f"prob_action_{i}" for i in range(n_act)]
    if not all(c in row.index for c in cols):
        return None
    return np.array([float(row[c]) for c in cols])


# ── Step selection ─────────────────────────────────────────────────────────────

def auto_select_step(df: pd.DataFrame, override: int | None) -> pd.Series:
    """
    Return the single log row for the highlighted step.

    Manual override: the row whose "step" column equals `override`.
    Auto-selection (match=True rows only):
        argmax(|Delta_resource| + |Delta_network| + |Delta_security|)
        -- the step where the agent's Q-value advantage/regret is largest
           in absolute magnitude across all three objectives simultaneously.
        For algorithms without diff columns (DQN, PPO, A2C), the first
        match=1 row is used.
    """
    if override is not None:
        matched = df[df["step"] == override]
        if not matched.empty:
            return matched.iloc[0]
        print(f"  [warn] step={override} not found; falling back to auto-selection.")

    adv = df[df["match"] == 1] if "match" in df.columns else df
    if adv.empty:
        adv = df

    diff_cols = ["weighted_resource_diff", "weighted_network_diff",
                 "weighted_security_diff"]
    available = [c for c in diff_cols if c in adv.columns]
    if not available:
        return adv.iloc[0]

    score = adv[available].abs().sum(axis=1)
    return adv.loc[score.idxmax()]


# ── Plot 1: Q-value landscape across all actions ──────────────────────────────
# Used by: Envelope, EUPG  (MORL algorithms with full per-action Q logging)

def plot_q_landscape(row: pd.Series, n_act: int, algo: str,
                     step_id: int, out_dir: str):
    """
    Grouped bar chart showing, for each action:
      - weighted Q per objective (3 bars per action, side-by-side)
      - scalarised Q as an overlaid line
    Actions are sorted by descending scalar_Q.
    Vertical dashed lines mark the executed action and the reference action.
    For EUPG the policy probability is overlaid as a scaled dashed line.
    """
    q_mat  = get_q_matrix(row, n_act)
    sq     = scalar_q(q_mat)
    probs  = get_probs(row, n_act)

    env_action = int(row["env_action"])
    ref_action = int(row["reference_action"])
    match      = int(row.get("match", -1))
    regime     = "Advantage" if match == 1 else "Regret"

    order = np.argsort(sq)[::-1]
    x = np.arange(n_act)
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax1 = plt.subplots(figsize=(max(10, n_act * 0.9), 5))
    ax2 = ax1.twinx()

    for obj_idx, (name, offset) in enumerate(zip(OBJ_NAMES, offsets)):
        vals = q_mat[order, obj_idx] * WEIGHTS[obj_idx]
        ax1.bar(x + offset, vals, width,
                label=f"w*Q_{name}",
                color=OBJ_COLORS[name], alpha=0.80, edgecolor="white", lw=0.4)

    ax2.plot(x, sq[order], color="black", linewidth=1.8,
             marker="D", markersize=5, label="Scalar Q (line)", zorder=10)

    # Policy probability overlay -- EUPG only
    if probs is not None:
        prob_sorted = probs[order]
        q_max = max(abs(sq)) if len(sq) > 0 else 1.0
        prob_scaled = prob_sorted * q_max
        ax2.plot(x, prob_scaled, color="crimson", linewidth=1.6,
                 linestyle="--", marker="o", markersize=5,
                 label="Policy prob (scaled)", zorder=9)
        for xi, p in zip(x, prob_sorted):
            ax2.text(xi, prob_scaled[xi] + 0.02 * q_max, f"{p:.4f}",
                     color="crimson", fontsize=6, ha="center")

    for a_sorted_pos, a_orig in enumerate(order):
        if a_orig == env_action:
            ax1.axvline(a_sorted_pos, color="navy", linestyle="--",
                        linewidth=1.4, alpha=0.8)
            ax1.text(a_sorted_pos + 0.05, ax1.get_ylim()[1] * 0.95,
                     f"exec\n(a{a_orig})", fontsize=7, color="navy", va="top")
        if a_orig == ref_action and a_orig != env_action:
            ax1.axvline(a_sorted_pos, color="darkred", linestyle=":",
                        linewidth=1.4, alpha=0.8)
            ax1.text(a_sorted_pos + 0.05, ax1.get_ylim()[1] * 0.85,
                     f"ref\n(a{a_orig})", fontsize=7, color="darkred", va="top")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"a{order[i]}" for i in range(n_act)], fontsize=8)
    ax1.set_xlabel("Action (sorted by scalar Q)", fontsize=9)
    ax1.set_ylabel("Weighted Q per objective", fontsize=9)
    ax2.set_ylabel("Scalar Q  /  Policy prob", fontsize=9)

    obj_patches = [Patch(facecolor=OBJ_COLORS[n], label=f"w*Q_{n}") for n in OBJ_NAMES]
    line_handles = [Line2D([0], [0], color="black", linewidth=1.8,
                            marker="D", markersize=5, label="Scalar Q")]
    if probs is not None:
        line_handles.append(Line2D([0], [0], color="crimson", linewidth=1.4,
                                   linestyle="--", marker="o", markersize=4,
                                   label="Policy prob"))
    ax1.legend(handles=obj_patches, loc="upper right", fontsize=7)
    ax2.legend(handles=line_handles, loc="upper center", fontsize=7)

    ax1.grid(True, alpha=0.25, axis="y")
    ax1.set_title(
        f"{algo}  -  Q-value Landscape at Step {step_id}  [{regime}]\n"
        f"Executed: a{env_action}  |  Reference: a{ref_action}  |  match={match}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_q_landscape.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Plot 2: Per-objective Q-diff (executed vs reference) ─────────────────────
# Used by: Envelope, EUPG  (MORL algorithms with weighted diff columns)

def plot_q_diff(row: pd.Series, n_act: int, algo: str,
                step_id: int, out_dir: str):
    """
    Bar chart: weighted Q-diff per objective between executed and reference
    action.  Green = gain, Red = loss.
    Annotated with match regime and action IDs.
    """
    diff_cols = ["weighted_resource_diff", "weighted_network_diff",
                 "weighted_security_diff"]
    available = [c for c in diff_cols if c in row.index]
    if not available:
        print(f"  [skip] {algo} step {step_id}: no diff columns.")
        return

    labels = [n for c, n in zip(diff_cols, OBJ_NAMES) if c in row.index]
    diffs  = [float(row[c]) for c in available]
    colors = ["#4CAF50" if d > 0 else "#F44336" for d in diffs]

    env_action = int(row["env_action"])
    ref_action = int(row["alt_action"])
    match      = int(row.get("match", -1))
    regime     = "Advantage (env==ref)" if match == 1 else "Regret (env!=ref)"

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, diffs, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7)

    for bar, val in zip(bars, diffs):
        offset = 0.02 * max(abs(v) for v in diffs) if diffs else 0.01
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (offset if val >= 0 else -offset),
                f"{val:.3f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("w*DeltaQ  (executed - reference)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(
        f"{algo}  -  Per-Objective Weighted Q-Diff at Step {step_id}\n"
        f"Executed: a{env_action}  vs  Reference: a{ref_action}  [{regime}]",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_q_diff.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Plot 3: State context (22 feat_* features) ────────────────────────────────
# Used by: ALL algorithms

def plot_state_context(row: pd.Series, algo: str,
                       step_id: int, out_dir: str):
    """
    Horizontal bar chart of all available feat_* features, colour-coded by
    category.  Generated for all algorithms; the executed action ID is
    annotated in the title.
    """
    available = [(col, lbl, cat)
                 for col, lbl, cat in FEAT_META
                 if col in row.index and pd.notna(row[col])]
    if not available:
        print(f"  [skip] {algo} step {step_id}: no feat_* columns.")
        return

    cols, lbls, cats = zip(*available)
    vals    = [float(row[c]) for c in cols]
    bcolors = [CATEGORY_COLORS[c] for c in cats]

    n = len(cols)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.38)))
    y = np.arange(n)
    ax.barh(y, vals, color=bcolors, alpha=0.82, edgecolor="white", lw=0.3)

    x_max = max(abs(v) for v in vals) if vals else 1.0
    for i, val in enumerate(vals):
        ax.text(val + 0.01 * x_max, i, f"{val:.3f}",
                va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Feature value at this step", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")

    cat_handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c)
                   for c in dict.fromkeys(cats)]
    ax.legend(handles=cat_handles, title="Category", fontsize=7,
              title_fontsize=8, loc="lower right", framealpha=0.85)

    env_action = int(row["env_action"])
    match      = int(row.get("match", -1))
    regime     = "Advantage" if match == 1 else "Regret"
    ax.set_title(
        f"{algo}  -  State Context at Step {step_id}  [{regime}]\n"
        f"Executed action: a{env_action}  |  {len(available)} features",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_state_context.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Plot 4: Pairwise RDX heatmap ─────────────────────────────────────────────
# Used by: Envelope, EUPG  (MORL algorithms with full per-action Q logging)

def plot_pairwise_rdx(row: pd.Series, n_act: int, algo: str,
                      step_id: int, out_dir: str):
    """
    Pairwise RDX for the highlighted step.

    For every ordered pair (a, b) with a != b:
        overall_adv(a, b) = scalar_Q_a - scalar_Q_b

    Displayed as an (n_act x n_act) heatmap: cell (a, b) = adv(a->b).
    Positive (green): a is preferred over b.
    Negative (red):   b is preferred over a.

    The executed and reference actions are highlighted with axis tick markers.
    Per-objective breakdown for the focal pair (executed vs reference) is shown
    in the right panel.
    """
    q_mat    = get_q_matrix(row, n_act)
    sq       = scalar_q(q_mat)
    wq       = q_mat * WEIGHTS

    env_action = int(row["env_action"])
    ref_action = int(row["alt_action"])
    match      = int(row.get("match", -1))
    regime     = "Advantage" if match == 1 else "Regret"

    adv_mat  = sq[:, None] - sq[None, :]
    diff_mat = wq[:, None, :] - wq[None, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    vmax = np.abs(adv_mat).max()
    im   = ax.imshow(adv_mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Scalar Q advantage  a->b")

    action_labels = [f"a{i}" for i in range(n_act)]
    ax.set_xticks(range(n_act))
    ax.set_yticks(range(n_act))
    ax.set_xticklabels(action_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(action_labels, fontsize=7)

    for a_id, color, lw in [(env_action, "navy", 2.5), (ref_action, "darkred", 1.8)]:
        ax.axhline(a_id - 0.5, color=color, lw=lw, alpha=0.7)
        ax.axhline(a_id + 0.5, color=color, lw=lw, alpha=0.7)
        ax.axvline(a_id - 0.5, color=color, lw=lw, alpha=0.7)
        ax.axvline(a_id + 0.5, color=color, lw=lw, alpha=0.7)

    ax.add_patch(plt.Rectangle(
        (ref_action - 0.5, env_action - 0.5), 1, 1,
        fill=False, edgecolor="black", lw=2.5, zorder=10,
    ))

    ax.set_xlabel("Action b  (opponent)", fontsize=9)
    ax.set_ylabel("Action a  (protagonist)", fontsize=9)
    ax.set_title(
        f"{algo}  -  Pairwise Q Advantage at Step {step_id}  [{regime}]\n"
        "cell (a,b) = scalar_Q(a) - scalar_Q(b)  >0: a is better",
        fontsize=9, fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color="navy", lw=2.5, label=f"Executed a{env_action}"),
        Line2D([0], [0], color="darkred", lw=1.8, label=f"Reference a{ref_action}"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
                       lw=2.5, label="Focal cell (exec->ref)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right", framealpha=0.85)

    ax2 = axes[1]
    focal_diff = diff_mat[env_action, ref_action]
    bcolors2   = ["#4CAF50" if d > 0 else "#F44336" for d in focal_diff]
    y2 = np.arange(3)
    ax2.barh(y2, focal_diff, color=bcolors2, alpha=0.85, edgecolor="black", lw=0.6)
    for i, val in enumerate(focal_diff):
        x_off = 0.02 * max(abs(v) for v in focal_diff) if focal_diff.any() else 0.01
        ax2.text(val + (x_off if val >= 0 else -x_off), i,
                 f"{val:.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=8, fontweight="bold")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(OBJ_NAMES, fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(0, color="black", lw=0.8, linestyle="--")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlabel("w*DeltaQ  (exec - ref)", fontsize=8)
    ax2.set_title(f"Focal pair\na{env_action}  vs  a{ref_action}",
                  fontsize=9, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_pairwise_rdx.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Correlation Plot A: Feature <-> Action Selection (ALL algorithms) ─────────

def plot_feat_action_corr(df: pd.DataFrame, n_act: int, algo: str,
                          step_id: int, out_dir: str):
    """
    Feature <-> Action Selection correlation heatmap.  Generated for ALL
    algorithms (value-based and policy-based alike).

    For each action i, a binary indicator column is constructed:
        selected_i = 1 if env_action == i else 0

    Pearson correlation is then computed between the feat_* state features and
    each binary indicator, answering: "which state features predict whether
    action i tends to be chosen?"

    Rows = feat_* features (up to 22)
    Cols = a0, a1, ..., a{n_act-1}  (binary selection indicators)

    Only columns with non-zero variance are included.
    """
    if "env_action" not in df.columns:
        print(f"  [skip] {algo}: no env_action column.")
        return

    feat_cols = [col for col, _, _ in FEAT_META if col in df.columns]
    if not feat_cols:
        print(f"  [skip] {algo}: no feature columns.")
        return

    action_mat = pd.DataFrame({
        f"a{i}": (df["env_action"] == i).astype(int)
        for i in range(n_act)
    })

    sub_feat   = df[feat_cols].select_dtypes(include="number")
    sub_feat   = sub_feat.loc[:, sub_feat.std() > 0]
    action_mat = action_mat.loc[:, action_mat.std() > 0]

    if sub_feat.empty or action_mat.empty:
        print(f"  [skip] {algo}: zero-variance columns, cannot compute correlation.")
        return

    corr = sub_feat.join(action_mat).corr().loc[sub_feat.columns, action_mat.columns]

    feat_lbl_map = {col: lbl for col, lbl, _ in FEAT_META}
    row_labels   = [feat_lbl_map.get(r, r) for r in corr.index]

    fig_h = max(6, len(row_labels) * 0.35)
    fig_w = max(8, len(corr.columns) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson correlation")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, fontsize=7, rotation=45)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(
        f"{algo}  -  Feature <-> Action Selection  (full log, {len(df):,} rows)\n"
        "Correlation between state features and binary action-selection indicators",
        fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_feat_action_corr.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Correlation Plot B: Feature <-> Q-value (VALUE-BASED only: DQN, Envelope) ─

def plot_feat_q_corr(df: pd.DataFrame, n_act: int, algo: str,
                     step_id: int, out_dir: str):
    """
    Feature <-> Q-value correlation heatmap.  Generated only for VALUE-BASED
    algorithms that log explicit per-action Q estimates (DQN, Envelope).

    Column formats supported:
      - q_a{i}_resource / _network / _security  ->  Envelope (MORL)
      - q_a{i}_scalar                           ->  DQN (single-objective)

    Rows = feat_* features (up to 22)
    Cols = Q columns for all logged actions.

    This reveals which state features are most strongly associated with high
    or low Q values across the full training log, providing global context
    for the single-step analysis.

    Only columns with non-zero variance are included.
    """
    feat_cols = [col for col, _, _ in FEAT_META if col in df.columns]
    q_cols    = []
    for i in range(n_act):
        for obj in ("resource", "network", "security", "scalar"):
            c = f"q_a{i}_{obj}"
            if c in df.columns:
                q_cols.append(c)

    if not feat_cols or not q_cols:
        print(f"  [skip] {algo}: feat/q columns missing for Q-value correlation plot.")
        return

    sub_feat = df[feat_cols].copy().select_dtypes(include="number")
    sub_feat = sub_feat.loc[:, sub_feat.std() > 0]
    sub_q    = df[q_cols].copy().select_dtypes(include="number")
    sub_q    = sub_q.loc[:, sub_q.std() > 0]

    if sub_feat.empty or sub_q.empty:
        print(f"  [skip] {algo}: all-zero-variance columns, cannot compute correlation.")
        return

    corr = sub_feat.join(sub_q).corr().loc[sub_feat.columns, sub_q.columns]

    # Shorten Q column labels: q_a3_resource -> a3_R, q_a3_scalar -> a3_Q
    short_cols = (corr.columns
                  .str.replace("q_a", "a", regex=False)
                  .str.replace("_resource", "_R", regex=False)
                  .str.replace("_network",  "_N", regex=False)
                  .str.replace("_security", "_S", regex=False)
                  .str.replace("_scalar",   "_Q", regex=False)
                  .tolist())
    feat_lbl_map = {col: lbl for col, lbl, _ in FEAT_META}
    short_rows   = [feat_lbl_map.get(r, r) for r in corr.index]

    fig_h = max(6, len(corr.index) * 0.35)
    fig_w = max(10, len(corr.columns) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson correlation")

    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, fontsize=5.5, rotation=90)
    ax.set_yticks(range(len(short_rows)))
    ax.set_yticklabels(short_rows, fontsize=7)
    ax.set_title(
        f"{algo}  -  Feature <-> Q-value Correlation  (full log, {len(df):,} rows)\n"
        "Correlation between state features and per-action Q-value estimates",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_feat_q_corr.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Correlation Plot C: Feature <-> Policy Probability (POLICY-BASED only) ────

def plot_feat_prob_corr(df: pd.DataFrame, n_act: int, algo: str,
                        step_id: int, out_dir: str):
    """
    Feature <-> Policy Probability correlation heatmap.  Generated only for
    POLICY-BASED algorithms that output an explicit action-probability
    distribution (PPO, A2C, EUPG).

    Rows = feat_* features (up to 22)
    Cols = prob_action_0, ..., prob_action_{n_act-1}

    This reveals which state features drive the actor's output probabilities,
    complementing the action-selection correlation with a softer, continuous
    signal that is not available for value-based algorithms.

    Only columns with non-zero variance are included.
    """
    prob_cols = [f"prob_action_{i}" for i in range(n_act)
                 if f"prob_action_{i}" in df.columns]
    if not prob_cols:
        print(f"  [skip] {algo}: no prob_action_* columns for probability correlation plot.")
        return

    feat_cols = [col for col, _, _ in FEAT_META if col in df.columns]
    if not feat_cols:
        print(f"  [skip] {algo}: no feature columns.")
        return

    sub_feat = df[feat_cols].select_dtypes(include="number")
    sub_feat = sub_feat.loc[:, sub_feat.std() > 0]
    sub_prob = df[prob_cols].select_dtypes(include="number")
    sub_prob = sub_prob.loc[:, sub_prob.std() > 0]

    if sub_feat.empty or sub_prob.empty:
        print(f"  [skip] {algo}: zero-variance columns, cannot compute correlation.")
        return

    corr = sub_feat.join(sub_prob).corr().loc[sub_feat.columns, sub_prob.columns]

    feat_lbl_map = {col: lbl for col, lbl, _ in FEAT_META}
    row_labels   = [feat_lbl_map.get(r, r) for r in corr.index]
    short_cols   = [c.replace("prob_action_", "a") for c in corr.columns]

    fig_h = max(6, len(row_labels) * 0.35)
    fig_w = max(8, len(short_cols) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson correlation")

    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, fontsize=7, rotation=45)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(
        f"{algo}  -  Feature <-> Policy Probability  (full log, {len(df):,} rows)\n"
        "Correlation between state features and actor output probabilities",
        fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    out = os.path.join(out_dir, f"{algo}_step{step_id}_feat_prob_corr.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")


# ── Summary statistics: advantage vs regret ───────────────────────────────────

def regime_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute per-algorithm, per-regime (advantage / regret) summary statistics
    for the three weighted Q-diff columns.
    Returns a DataFrame saved to {out}/regime_summary.csv.
    """
    rows = []
    for algo, df in dfs.items():
        if df is None or df.empty:
            continue
        adv  = df[df["match"] == 1] if "match" in df.columns else pd.DataFrame()
        regr = df[df["match"] == 0] if "match" in df.columns else pd.DataFrame()

        for regime, sub in [("Advantage (match=1)", adv),
                             ("Regret    (match=0)", regr)]:
            if sub.empty:
                continue
            row_d = {
                "Algorithm": algo,
                "Regime":    regime,
                "N_steps":   len(sub),
                "Match_rate_%": (f"{len(adv)/(len(adv)+len(regr))*100:.1f}"
                                 if (len(adv)+len(regr)) > 0 else "-"),
            }
            for col, label in [
                ("weighted_resource_diff", "Mean_Delta_Resource"),
                ("weighted_network_diff",  "Mean_Delta_Network"),
                ("weighted_security_diff", "Mean_Delta_Security"),
            ]:
                if col in sub.columns:
                    row_d[label]             = f"{sub[col].mean():.4f}"
                    row_d[label + "_std"]    = f"{sub[col].std():.4f}"
                    row_d[label + "_median"] = f"{sub[col].median():.4f}"
            rows.append(row_d)

    return pd.DataFrame(rows)


# ── Top-k pairwise ranking table ──────────────────────────────────────────────

def print_pairwise_ranking(row: pd.Series, n_act: int, algo: str,
                           step_id: int, top_k: int = 10):
    """
    Print the top-k and bottom-k action pairs ranked by overall advantage
    (scalar_Q_a - scalar_Q_b) for the highlighted step.
    Marks the focal pair (executed vs reference) with a star symbol.
    Used by MORL algorithms (Envelope, EUPG) that log full Q matrices.
    """
    q_mat = get_q_matrix(row, n_act)
    sq    = scalar_q(q_mat)
    wq    = q_mat * WEIGHTS

    env_action = int(row["env_action"])
    ref_action = int(row["reference_action"])

    records = []
    for a, b in itertools.permutations(range(n_act), 2):
        diff    = wq[a] - wq[b]
        overall = float(sq[a] - sq[b])
        records.append({
            "action_a":    a,
            "action_b":    b,
            "Delta_R":     f"{diff[0]:+.3f}",
            "Delta_N":     f"{diff[1]:+.3f}",
            "Delta_S":     f"{diff[2]:+.3f}",
            "Overall_Adv": overall,
            "focal":       "*" if a == env_action and b == ref_action else "",
        })

    ranked = sorted(records, key=lambda r: r["Overall_Adv"], reverse=True)
    print(f"\n  -- {algo} Step {step_id}: Pairwise ranking (top {top_k} / bottom {top_k}) --")
    header = f"  {'a->b':>6}  {'Delta_R':>9}  {'Delta_N':>9}  {'Delta_S':>9}  {'Overall':>9}  {'':>2}"
    print(header)
    print("  " + "-" * 58)
    for rec in ranked[:top_k]:
        print(f"  a{rec['action_a']}->a{rec['action_b']:>2}  "
              f"{rec['Delta_R']:>9}  {rec['Delta_N']:>9}  "
              f"{rec['Delta_S']:>9}  {rec['Overall_Adv']:>+9.3f}  "
              f"{rec['focal']:>2}")
    print("  ...")
    for rec in ranked[-top_k:]:
        print(f"  a{rec['action_a']}->a{rec['action_b']:>2}  "
              f"{rec['Delta_R']:>9}  {rec['Delta_N']:>9}  "
              f"{rec['Delta_S']:>9}  {rec['Overall_Adv']:>+9.3f}  "
              f"{rec['focal']:>2}")


# ── Per-step execution analysis text ─────────────────────────────────────────

def print_step_analysis(row: pd.Series, n_act: int, algo: str):
    """
    Print a structured text summary of the highlighted step.

    For MORL algorithms (Envelope, EUPG) this includes the full scalar Q
    ranking and weighted Q-diff breakdown.
    For single-objective algorithms (DQN, PPO, A2C) the Q / prob section is
    printed only when the relevant columns are present.
    """
    q_mat  = get_q_matrix(row, n_act)
    sq     = scalar_q(q_mat)
    wq     = q_mat * WEIGHTS
    probs  = get_probs(row, n_act)

    env_action = int(row["env_action"])
    ref_action = int(row["reference_action"])
    match      = int(row.get("match", -1))
    step_id    = int(row["step"])
    regime     = "ADVANTAGE (match=1)" if match == 1 else "REGRET (match=0)"

    print(f"\n{'='*65}")
    print(f"  {algo}  |  Step {step_id}  |  Regime: {regime}")
    print(f"{'='*65}")
    print(f"  Executed action  : a{env_action}")
    print(f"  Reference action : a{ref_action}  "
          f"({'argmax scalar Q' if algo in VALUE_BASED_ALGOS else 'argmax prob'})")
    print(f"  Match            : {match}  "
          f"({'env chose the reference action' if match else 'env deviated from reference'})")
    print()

    # Scalar Q ranking -- meaningful for DQN and MORL algorithms
    if n_act > 0 and any(sq != 0):
        print("  Scalar Q ranking (all actions):")
        order = np.argsort(sq)[::-1]
        for rank, a in enumerate(order):
            marker = " <- exec" if a == env_action else (
                     " <- ref " if a == ref_action else "")
            print(f"    Rank {rank+1:2d}:  a{a:2d}  scalar_Q = {sq[a]:+9.3f}{marker}")

    # Policy probability -- policy-based algorithms (PPO, A2C, EUPG)
    if probs is not None:
        print("\n  Policy probabilities (all actions):")
        porder = np.argsort(probs)[::-1]
        for rank, a in enumerate(porder):
            marker = " <- exec" if a == env_action else (
                     " <- ref " if a == ref_action else "")
            print(f"    Rank {rank+1:2d}:  a{a:2d}  prob = {probs[a]:.6f}{marker}")

    # Weighted Q-diff -- MORL algorithms only (diff columns present)
    diff_cols_present = any(c in row.index for c in [
        "weighted_resource_diff", "weighted_network_diff", "weighted_security_diff"])
    if diff_cols_present and n_act > 0 and any(sq != 0):
        print(f"\n  Weighted Q-diff  (exec a{env_action} - ref a{ref_action}):")
        for obj, w, name in zip(range(3), WEIGHTS, OBJ_NAMES):
            d = float(wq[env_action, obj] - wq[ref_action, obj])
            arrow = "^" if d > 0 else "v"
            print(f"    {name:>10}: {d:+9.3f}  {arrow}")
        overall = float(sq[env_action] - sq[ref_action])
        print(f"    {'Overall':>10}: {overall:+9.3f}")
    print()

    # State context highlights (top 5 by absolute value) -- all algorithms
    feat_avail = [(col, lbl, cat)
                  for col, lbl, cat in FEAT_META
                  if col in row.index and pd.notna(row[col])]
    if feat_avail:
        vals_abs = [(abs(float(row[col])), col, lbl, cat)
                    for col, lbl, cat in feat_avail]
        top5 = sorted(vals_abs, reverse=True)[:5]
        print("  Top-5 state features by absolute magnitude:")
        for _, col, lbl, cat in top5:
            print(f"    [{cat:>13}]  {lbl:<28} = {float(row[col]):.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-Step RDX Analysis for RL / MORL algorithms.\n"
            "Supports value-based (DQN, Envelope) and policy-based (PPO, A2C, EUPG).\n\n"
            "Plot routing per algorithm:\n"
            "  Envelope / EUPG : q_landscape, q_diff, pairwise_rdx, state_context,\n"
            "                    feat_action_corr\n"
            "                    + feat_q_corr (Envelope) / feat_prob_corr (EUPG)\n"
            "  DQN             : state_context, feat_action_corr, feat_q_corr\n"
            "  PPO / A2C       : state_context, feat_action_corr, feat_prob_corr"
        )
    )
    parser.add_argument("--envelope",      default=None,
                        help="Path to Envelope explain CSV  [value-based, MORL]")
    parser.add_argument("--eupg",          default=None,
                        help="Path to EUPG explain CSV  [policy-based, MORL]")
    parser.add_argument("--dqn",           default=None,
                        help="Path to DQN explain CSV  [value-based, single-objective]")
    parser.add_argument("--ppo",           default=None,
                        help="Path to PPO explain CSV  [policy-based, single-objective]")
    parser.add_argument("--a2c",           default=None,
                        help="Path to A2C explain CSV  [policy-based, single-objective]")
    parser.add_argument("--step",          type=int, default=None,
                        help="Manually pin the step to highlight (all algos). "
                             "Omit for auto-selection.")
    parser.add_argument("--step-envelope", type=int, default=None,
                        help="Step override for Envelope only.")
    parser.add_argument("--step-eupg",     type=int, default=None,
                        help="Step override for EUPG only.")
    parser.add_argument("--step-dqn",      type=int, default=None,
                        help="Step override for DQN only.")
    parser.add_argument("--step-ppo",      type=int, default=None,
                        help="Step override for PPO only.")
    parser.add_argument("--step-a2c",      type=int, default=None,
                        help="Step override for A2C only.")
    parser.add_argument("--out",           default="./single_step_plots",
                        help="Output directory for png plots")
    parser.add_argument("--top-k",         type=int, default=10,
                        help="Top-k pairs to print in pairwise ranking (MORL only)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n-- Loading CSV files --")
    dfs = {}
    if args.envelope: dfs["Envelope"] = load_csv(args.envelope, "Envelope")
    if args.eupg:     dfs["EUPG"]     = load_csv(args.eupg,     "EUPG")
    if args.dqn:      dfs["DQN"]      = load_csv(args.dqn,      "DQN")
    if args.ppo:      dfs["PPO"]      = load_csv(args.ppo,      "PPO")
    if args.a2c:      dfs["A2C"]      = load_csv(args.a2c,      "A2C")

    if not dfs:
        print("No CSV files loaded. Provide at least one algorithm flag.")
        sys.exit(1)

    # Per-algorithm step overrides
    step_overrides = {
        "Envelope": args.step_envelope,
        "EUPG":     args.step_eupg,
        "DQN":      args.step_dqn,
        "PPO":      args.step_ppo,
        "A2C":      args.step_a2c,
    }

    # ── Per-algorithm analysis ─────────────────────────────────────────────
    for algo, df in dfs.items():
        if df is None or df.empty:
            continue

        print(f"\n-- {algo} --")

        step_override = step_overrides.get(algo) or args.step
        row     = auto_select_step(df, step_override)
        step_id = int(row["step"])
        n_act   = n_actions(df)

        print(f"  Highlighted step: {step_id}  (n_actions = {n_act})")

        # Text analysis (adapts output to available columns)
        print_step_analysis(row, n_act, algo)

        # ── png plots (routing by algorithm) ──────────────────────────────
        print(f"\n  Generating plots -> {args.out}/")

        # State context -- ALL algorithms
        plot_state_context(row, algo, step_id, args.out)

        # Feature <-> Action Selection -- ALL algorithms
        plot_feat_action_corr(df, n_act, algo, step_id, args.out)

        # MORL-only plots: Q-landscape, Q-diff, pairwise RDX, pairwise ranking
        if algo in MORL_ALGOS:
            plot_q_landscape(row, n_act, algo, step_id, args.out)
            plot_q_diff(row, n_act, algo, step_id, args.out)
            plot_pairwise_rdx(row, n_act, algo, step_id, args.out)
            print_pairwise_ranking(row, n_act, algo, step_id, top_k=args.top_k)

        # Feature <-> Q-value -- value-based only (DQN, Envelope)
        if algo in VALUE_BASED_ALGOS:
            plot_feat_q_corr(df, n_act, algo, step_id, args.out)

        # Feature <-> Policy Probability -- policy-based only (PPO, A2C, EUPG)
        if algo in POLICY_BASED_ALGOS:
            plot_feat_prob_corr(df, n_act, algo, step_id, args.out)

    # ── Cross-algorithm regime summary ─────────────────────────────────────
    print("\n-- Regime summary (advantage vs regret) --")
    summary = regime_summary(dfs)
    if not summary.empty:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary.to_string(index=False))
        out_csv = os.path.join(args.out, "regime_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"\n  Regime summary saved -> {out_csv}")

    print(f"\nAll outputs saved in: {args.out}/")


if __name__ == "__main__":
    main()