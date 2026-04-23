"""
Cross-Algorithm RDX Evaluation Script
=====================================
Evaluates and compares RDX explanation logs for 5 RL algorithms:
  - A2C      (SingleObjective_PolicyBased, q_type=A2C_Q)
  - DQN      (SingleObjective_RDX,         q_type=Scalar)
  - Envelope (MORL_RDX,                    q_type=MORL_vec)
  - EUPG     (MORL_RDX,                    q_type=MORL_vec)
  - PPO      (SingleObjective_PolicyBased, q_type=PPO_Q)

Delta semantics (IMPORTANT):
  - match=True  (env_action == reference_action):
        delta = Q(best_action) - Q(next_best_action)
        → "advantage": how much does the agent prefer its chosen action over the second-best option?
  - match=False (env_action != reference_action):
        delta = Q(env_action) - Q(reference_action)
        → "regret"

  These two quantities have DIFFERENT meanings, so they are
  NEVER mixed in the same plot or summary statistic.

Usage:
  Edit the CSV_FILES dict below, then run:
      python evaluate_all_algorithms.py

Outputs:
  - Per-algorithm plots  (PDF in OUTPUT_DIR)
  - Cross-algorithm comparison plots (PDF in OUTPUT_DIR)
  - Summary table printed to stdout and saved as CSV
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── 0. CONFIGURATION ──────────────────────────────────────────────────────────

CSV_FILES = {
    "A2C":      "a2c_explain.csv",
    "DQN":      "dqn_explain.csv",
    "Envelope": "envelope_explain.csv",
    "EUPG":     "eupg_explain.csv",
    "PPO":      "ppo_explain.csv",
}

WEIGHTS     = np.array([0.4, 0.3, 0.3])
OBJ_NAMES   = ["Resource", "Network", "Security"]
COLORS_OBJ  = {"Resource": "steelblue", "Network": "darkorange", "Security": "green"}
ALGO_COLORS = {
    "A2C":      "#e41a1c",
    "DQN":      "#377eb8",
    "Envelope": "#4daf4a",
    "EUPG":     "#984ea3",
    "PPO":      "#ff7f00",
}
OUTPUT_DIR  = "rdx_evaluation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_FRAC = 0.02   # rolling window as fraction of total steps

# ── 1. HELPERS ────────────────────────────────────────────────────────────────

MORL_ALGOS   = {"Envelope", "EUPG"}
SCALAR_ALGOS = {"A2C", "DQN", "PPO"}

def is_morl(algo: str) -> bool:
    return algo in MORL_ALGOS

def load_csv(path: str) -> pd.DataFrame | None:
    if path is None or not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def window(n: int) -> int:
    return max(1, int(n * WINDOW_FRAC))

def split_by_match(df: pd.DataFrame):
    """
    Return (df_advantage, df_regret).

    df_advantage : rows where env_action == reference_action
        delta = Q(best) - Q(next_best)  >=0, measures decisiveness
    df_regret    : rows where env_action != reference_action
        delta = Q(env_action) - Q(best) <=0, measures execution cost
    """
    if "match" not in df.columns:
        # Infer match from action columns if possible
        if "env_action" in df.columns and "reference_action" in df.columns:
            df = df.copy()
            df["match"] = (df["env_action"] == df["reference_action"])
        else:
            return df.copy(), pd.DataFrame(columns=df.columns)
    adv  = df[df["match"] == True].copy()
    regr = df[df["match"] == False].copy()
    return adv, regr

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────

dfs: dict[str, pd.DataFrame] = {}
for algo, path in CSV_FILES.items():
    df = load_csv(path)
    if df is not None:
        dfs[algo] = df
        print(f"[✓] Loaded {algo}: {len(df):,} rows  →  {path}")
    else:
        print(f"[–] Skipped {algo}: file not found ({path})")

if not dfs:
    raise RuntimeError("No CSV files loaded. Check CSV_FILES paths.")

# ── 3. PER-ALGORITHM PLOTS ────────────────────────────────────────────────────

# ── 3a. Action distribution ───────────────────────────────────────────────────

def plot_action_distribution(algo: str, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{algo} – MTD Action Selection Distribution",
                 fontsize=13, fontweight="bold")

    all_actions = sorted(set(
        df["env_action"].dropna().unique().tolist() +
        df["reference_action"].dropna().unique().tolist()
    ))
    cmap   = plt.cm.tab20(np.linspace(0, 1, max(len(all_actions), 1)))
    acolor = {a: cmap[i] for i, a in enumerate(all_actions)}

    for ax, col, title in zip(
        axes,
        ["reference_action", "env_action"],
        ["Explained Best Action (reference)", "Actual Executed Action (env)"]
    ):
        counts = df[col].value_counts().sort_index()
        colors = [acolor.get(a, "gray") for a in counts.index]
        bars   = ax.bar(counts.index.astype(str), counts.values, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Action ID")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(val),
                    ha="center", va="bottom", fontsize=7)

    if "match" in df.columns:
        mr = df["match"].mean() * 100
        fig.text(0.5, 0.01,
                 f"Match Rate (env == reference): {mr:.1f}%",
                 ha="center", fontsize=10, color="darkred", fontweight="bold")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUTPUT_DIR, f"{algo}_action_distribution.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 3b. Scalar delta — split into advantage & regret ─────────────────────────

def plot_scalar_delta_split(algo: str, df: pd.DataFrame):
    """
    Two subplots side-by-side:
      Left  – Advantage (match=True):  rolling mean of delta (>=0)
      Right – Regret    (match=False): rolling mean of delta (<=0)
    Each subplot also shows a ±1-std band and a step-count subtitle.
    """
    if "delta" not in df.columns:
        return

    adv, regr = split_by_match(df)
    color = ALGO_COLORS.get(algo, "gray")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{algo} – Q-value Delta Split by Match Status",
                 fontsize=13, fontweight="bold")

    for ax, subset, label, subtitle, ylab in [
        (axes[0], adv,  "Advantage (env==ref)",
         f"Steps: {len(adv):,}  |  env_action == reference_action",
         "Q(best) − Q(next-best)  ≥ 0"),
        (axes[1], regr, "Regret (env≠ref)",
         f"Steps: {len(regr):,}  |  env_action ≠ reference_action",
         "Q(env) − Q(best)  ≤ 0"),
    ]:
        if subset.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            ax.set_title(f"{label}\n({subtitle})", fontsize=9)
            continue
        w = window(len(subset))
        rolling_mean = subset["delta"].rolling(w).mean()
        rolling_std  = subset["delta"].rolling(w).std()
        ax.plot(subset["step"], rolling_mean,
                color=color, linewidth=1.5, label=label)
        ax.fill_between(subset["step"],
                         rolling_mean - rolling_std,
                         rolling_mean + rolling_std,
                         color=color, alpha=0.15, label="±1 std")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{label}\n{subtitle}", fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylab, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{algo}_delta_split.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 3c. MORL delta — split into advantage & regret ───────────────────────────

MORL_DIFF_COLS = [
    "weighted_resource_diff",
    "weighted_network_diff",
    "weighted_security_diff",
]

def plot_morl_delta_split(algo: str, df: pd.DataFrame):
    """
    Two rows × 3 columns:
      Top row    – Advantage (match=True)
      Bottom row – Regret    (match=False)
    Each cell = one objective's rolling mean delta.
    """
    available = [c for c in MORL_DIFF_COLS if c in df.columns]
    if not available:
        return

    adv, regr = split_by_match(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey="row")
    fig.suptitle(f"{algo} – Weighted Objective Delta Split by Match Status",
                 fontsize=13, fontweight="bold")

    row_meta = [
        (adv,  "Advantage  (env == reference): Q(best)−Q(next-best) ≥ 0"),
        (regr, "Regret     (env ≠  reference): Q(env)−Q(best) ≤ 0"),
    ]
    for row_idx, (subset, row_label) in enumerate(row_meta):
        for col_idx, (col, name) in enumerate(zip(MORL_DIFF_COLS, OBJ_NAMES)):
            ax = axes[row_idx][col_idx]
            if col not in df.columns:
                ax.set_visible(False)
                continue
            if subset.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
            else:
                w = window(len(subset))
                rm = subset[col].rolling(w).mean()
                rs = subset[col].rolling(w).std()
                ax.plot(subset["step"], rm,
                        color=COLORS_OBJ[name], linewidth=1.5, label=name)
                ax.fill_between(subset["step"], rm - rs, rm + rs,
                                color=COLORS_OBJ[name], alpha=0.15)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlabel("Step", fontsize=8)
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(f"W-Δ  |  {row_label}", fontsize=7.5)
            title_line = f"{name}"
            if not subset.empty:
                title_line += f"  ({len(subset):,} steps)"
            ax.set_title(title_line, fontsize=9)
            ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{algo}_morl_delta_split.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 3d. MORL pairwise scatter — split ────────────────────────────────────────

def plot_morl_pairwise_split(algo: str, df: pd.DataFrame):
    """
    Pairwise scatter for MORL objectives.
    Left column = Advantage (match=True), Right column = Regret (match=False).
    """
    pairs = [
        ("weighted_resource_diff", "weighted_network_diff",  "W-Resource", "W-Network"),
        ("weighted_resource_diff", "weighted_security_diff", "W-Resource", "W-Security"),
        ("weighted_network_diff",  "weighted_security_diff", "W-Network",  "W-Security"),
    ]
    pairs = [(x, y, xl, yl) for x, y, xl, yl in pairs
             if x in df.columns and y in df.columns]
    if not pairs:
        return

    adv, regr = split_by_match(df)
    unique_actions = sorted(df["reference_action"].dropna().unique())
    cmap   = plt.cm.tab20(np.linspace(0, 1, max(len(unique_actions), 1)))
    acolor = {a: cmap[i] for i, a in enumerate(unique_actions)}

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"{algo} – Pairwise Objective Scatter Split by Match Status",
                 fontsize=13, fontweight="bold")

    col_titles = [
        "Advantage  (env == ref)",
        "Regret     (env ≠  ref)",
    ]
    for col_idx, (subset, ct) in enumerate(zip([adv, regr], col_titles)):
        for row_idx, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
            ax = axes[row_idx][col_idx]
            if subset.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
            else:
                pcolors = [acolor.get(a, "gray") for a in subset["reference_action"]]
                ax.scatter(subset[xcol], subset[ycol],
                           c=pcolors, alpha=0.5, s=15, edgecolors="none")
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(ct, fontsize=10, fontweight="bold")

    legend_handles = [Patch(facecolor=acolor[a], label=f"Action {a}")
                      for a in unique_actions]
    fig.legend(handles=legend_handles, title="Best Action ID",
               bbox_to_anchor=(1.01, 0.5), loc="center left",
               fontsize=7, title_fontsize=8, framealpha=0.8)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    out = os.path.join(OUTPUT_DIR, f"{algo}_pairwise_scatter_split.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 3e. Cumulative delta — split ──────────────────────────────────────────────

def plot_cumulative_delta_split(algo: str, df: pd.DataFrame):
    """
    Two subplots:
      Left  – cumulative advantage delta  (match=True rows)
      Right – cumulative regret delta     (match=False rows)
    MORL: one line per objective.  Scalar: single line.
    """
    adv, regr = split_by_match(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{algo} – Cumulative Delta Split by Match Status",
                 fontsize=13, fontweight="bold")

    subtitles = [
        ("Advantage  (env == ref)\nQ(best)−Q(next-best)", adv),
        ("Regret     (env ≠  ref)\nQ(env)−Q(best)",       regr),
    ]

    for ax, (subtitle, subset) in zip(axes, subtitles):
        ax.set_title(subtitle, fontsize=9)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        if subset.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            continue

        if is_morl(algo):
            for col, name in zip(MORL_DIFF_COLS, OBJ_NAMES):
                if col not in subset.columns:
                    continue
                ax.plot(subset["step"], subset[col].cumsum(),
                        label=name, color=COLORS_OBJ[name], linewidth=1.5)
            ax.set_ylabel("Cumulative Weighted Q-diff")
        else:
            if "delta" not in subset.columns:
                ax.text(0.5, 0.5, "No delta column",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="gray")
                continue
            ax.plot(subset["step"], subset["delta"].cumsum(),
                    color=ALGO_COLORS.get(algo, "gray"), linewidth=1.5)
            ax.set_ylabel("Cumulative Q-diff")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{algo}_cumulative_delta_split.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 3f. Policy entropy (EUPG) ────────────────────────────────────────────────

def plot_action_entropy_if_available(algo: str, df: pd.DataFrame):
    if "policy_entropy" not in df.columns:
        return
    w = window(len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["policy_entropy"].rolling(w).mean(),
            color=ALGO_COLORS.get(algo, "gray"), linewidth=1.5)
    ax.set_title(f"{algo} – Policy Entropy Over Time")
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{algo}_entropy.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


print("\n── Per-algorithm plots ──")
for algo, df in dfs.items():
    print(f"\n[{algo}]")
    plot_action_distribution(algo, df)
    if is_morl(algo):
        plot_morl_delta_split(algo, df)
        plot_morl_pairwise_split(algo, df)
    else:
        plot_scalar_delta_split(algo, df)
    plot_cumulative_delta_split(algo, df)
    plot_action_entropy_if_available(algo, df)


# ── 4. CROSS-ALGORITHM COMPARISON ────────────────────────────────────────────

print("\n── Cross-algorithm comparison plots ──")

# ── 4a. Match rate ────────────────────────────────────────────────────────────

algos_with_match = [a for a, df in dfs.items() if "match" in df.columns]
if algos_with_match:
    match_rates = {a: dfs[a]["match"].mean() * 100 for a in algos_with_match}
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(match_rates.keys()),
                  list(match_rates.values()),
                  color=[ALGO_COLORS.get(a, "gray") for a in match_rates])
    for bar, (algo, mr) in zip(bars, match_rates.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{mr:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_title("Match Rate Comparison (env_action == reference_action)")
    ax.set_ylabel("Match Rate (%)")
    ax.set_xlabel("Algorithm")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparison_match_rate.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 4b. Scalar advantage comparison (match=True only) ────────────────────────

scalar_loaded = [a for a in SCALAR_ALGOS if a in dfs and "delta" in dfs[a].columns]
if len(scalar_loaded) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Scalar Algorithms – Delta Comparison Split by Match Status",
                 fontsize=13, fontweight="bold")

    for ax, use_match, title_suffix, ylabel in [
        (axes[0], True,
         "Advantage  (env == ref) — Q(best)−Q(next-best)",
         "Q-value Advantage  ≥ 0"),
        (axes[1], False,
         "Regret     (env ≠  ref) — Q(env)−Q(best)",
         "Q-value Regret  ≤ 0"),
    ]:
        for algo in scalar_loaded:
            df      = dfs[algo]
            adv, regr = split_by_match(df)
            subset  = adv if use_match else regr
            if subset.empty:
                continue
            w = window(len(subset))
            ax.plot(subset["step"], subset["delta"].rolling(w).mean(),
                    label=algo, color=ALGO_COLORS.get(algo, "gray"), linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title_suffix, fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparison_scalar_delta_split.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 4c. MORL per-objective comparison — split ─────────────────────────────────

morl_loaded = [a for a in MORL_ALGOS if a in dfs]
if len(morl_loaded) >= 2:
    for use_match, fname_suffix, row_label in [
        (True,  "advantage",
         "Advantage (env == ref): Q(best)−Q(next-best)  ≥ 0"),
        (False, "regret",
         "Regret    (env ≠  ref): Q(env)−Q(best)  ≤ 0"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        fig.suptitle(f"MORL Weighted Objective Delta – {row_label}",
                     fontsize=12, fontweight="bold")
        for ax, col, name in zip(axes, MORL_DIFF_COLS, OBJ_NAMES):
            for algo in morl_loaded:
                df = dfs[algo]
                adv, regr = split_by_match(df)
                subset = adv if use_match else regr
                if col not in df.columns or subset.empty:
                    continue
                w = window(len(subset))
                ax.plot(subset["step"], subset[col].rolling(w).mean(),
                        label=algo,
                        color=ALGO_COLORS.get(algo, "gray"), linewidth=1.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Weighted Q-diff (rolling avg)", fontsize=9)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR,
                           f"comparison_morl_per_objective_{fname_suffix}.pdf")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ── 4d. Action distribution overlap ──────────────────────────────────────────

if len(dfs) >= 2:
    fig, ax = plt.subplots(figsize=(10, 4))
    for algo, df in dfs.items():
        counts = df["env_action"].value_counts().sort_index()
        total  = counts.sum()
        ax.plot(counts.index.astype(str),
                (counts.values / total) * 100,
                marker="o", markersize=4, label=algo,
                color=ALGO_COLORS.get(algo, "gray"), linewidth=1.5)
    ax.set_title("Action Selection Distribution Comparison (env_action, % of steps)")
    ax.set_xlabel("Action ID"); ax.set_ylabel("% of steps")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparison_action_distribution.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 5. SUMMARY TABLE ──────────────────────────────────────────────────────────
#
# For each algorithm we report:
#   Match Rate          : % steps where env_action == reference_action
#   Advantage Mean Δ    : mean delta on match=True rows  (>=0, decisiveness)
#   Regret    Mean Δ    : mean delta on match=False rows (<=0, execution cost)
#
# For MORL: Advantage/Regret are reported PER objective (weighted).
# These two columns are NEVER averaged together.
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Summary Table ──")
rows = []
for algo, df in dfs.items():
    adv, regr = split_by_match(df)
    total = len(df)

    row = {
        "Algorithm":         algo,
        "Total Steps":       total,
        "Match Rate (%)":    f"{df['match'].mean()*100:.1f}" if "match" in df.columns else "N/A",
        "Adv Steps":         len(adv),
        "Regret Steps":      len(regr),
    }

    if is_morl(algo):
        for col, name in zip(MORL_DIFF_COLS, OBJ_NAMES):
            if col not in df.columns:
                continue
            row[f"Adv Mean W-{name}Δ"]  = (
                f"{adv[col].mean():.4f}" if not adv.empty  else "—"
            )
            row[f"Regret Mean W-{name}Δ"] = (
                f"{regr[col].mean():.4f}" if not regr.empty else "—"
            )
    else:
        if "delta" in df.columns:
            row["Adv Mean Q-Δ"]    = f"{adv['delta'].mean():.4f}"  if not adv.empty  else "—"
            row["Regret Mean Q-Δ"] = f"{regr['delta'].mean():.4f}" if not regr.empty else "—"

    rows.append(row)

summary = pd.DataFrame(rows).set_index("Algorithm")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print(summary.to_string())

summary_out = os.path.join(OUTPUT_DIR, "summary_table.csv")
summary.to_csv(summary_out)
print(f"\nSummary table saved to: {summary_out}")
print(f"All plots saved in:      {OUTPUT_DIR}/")