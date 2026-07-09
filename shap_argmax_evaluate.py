"""
SHAP Analysis Visualization & Comparison
=========================================
Reads the SHAP output CSVs produced by shap_explain.py and generates:

  4.  Customized SHAP — per-algo summary bar chart (signed mean SHAP)
  5.  Customized SHAP — cross-algo radar chart (top shared features)
  6.  Envelope only  — objective influence pie + bar
  7.  Envelope only  — per-objective feature importance comparison
  8.  Cross-algo     — rank-based feature agreement matrix (Spearman)
  9.  Cross-algo     — feature importance divergence heatmap
  11. Dashboard      — top-5 features per algo side-by-side

Feature scope (aligned with shap_explain.py)
---------------------------------------------
Only the 5 objective-related features are used:
    feat_mean_mtd_overhead        (Resource)
    feat_mean_network_penalty     (Network)
    feat_max_network_penalty      (Network)
    feat_mean_security_penalty    (Security)
    feat_max_security_penalty     (Security)

File layout expected (each file under data_root/<algo>/<fname>, see _path)
----------------------------------------------------------------------------
Cross-algo comparison plots (4, 5, 8, 9, 11) — one "chosen-action Φ_s"
summary per algo, all structurally equivalent (mean |SHAP| / mean SHAP
for the action the policy actually selects):
    dqn/shap_dqn_scalar_Q_summary.csv
    envelope/shap_envelope_scalar_Q_summary.csv
    eupg/shap_eupg_policy_prob_summary.csv
    ppo/shap_ppo_policy_prob_summary.csv
    a2c/shap_a2c_policy_prob_summary.csv

Envelope-only diagnostic plots (6, 7) — per-objective Φ_s, NOT used in the
cross-algo plots above:
    envelope/shap_envelope_Q_resource_summary.csv
    envelope/shap_envelope_Q_network_summary.csv
    envelope/shap_envelope_Q_security_summary.csv
    envelope/shap_envelope_objective_influence.csv

Usage
-----
  python shap_evaluate.py --data_root ./shap_outputs --out_dir ./figures
  python shap_evaluate.py --data_root ./shap_outputs --out_dir ./figures --plot 4,6,11
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# ── palette ────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "dqn":      "#E07B54",
    "envelope": "#5B8DB8",
    "eupg":     "#6BAE75",
    "ppo":      "#A97DC4",
    "a2c":      "#E8C84A",
}
OBJ_COLORS = {
    "resource": "#E07B54",
    "network":  "#5B8DB8",
    "security": "#6BAE75",
}
ALGOS = ["dqn", "envelope", "eupg", "ppo", "a2c"]

# 5 objective-related features only — must match shap_explain.py FEATURE_COLS
# FEATURE_COLS = [
#     "feat_mean_mtd_overhead",
#     "feat_mean_network_penalty",
#     "feat_max_network_penalty",
#     "feat_mean_security_penalty",
#     "feat_max_security_penalty",
# ]
# SHORT_NAMES = {
#     "feat_mean_mtd_overhead":       "MTD\nOverhead",
#     "feat_mean_network_penalty":    "Net Penalty\n(mean)",
#     "feat_max_network_penalty":     "Net Penalty\n(max)",
#     "feat_mean_security_penalty":   "Sec Penalty\n(mean)",
#     "feat_max_security_penalty":    "Sec Penalty\n(max)",
# }

FEATURE_COLS = [
    # --- Security ---
    "feat_max_apt_score",           # apt cvss/asp score 
    "feat_mean_apt_score",
    "feat_max_dataleak_score",      # data_leak cvss/asp score 
    "feat_mean_dataleak_score",
    "feat_max_dos_score",           # dos cvss/asp score 
    "feat_mean_dos_score",

    # --- Resource ---
    "feat_vim0_cpu",               
    "feat_vim0_ram",
    "feat_vim1_cpu",
    "feat_vim1_ram",
    "feat_mean_remaining_mig",
    "feat_mean_remaining_reinst",

    # --- Network ---
    "feat_total_ues",              
]

SHORT_NAMES = {
    # --- Security ---
    "feat_max_apt_score":            "APT\n(max)",
    "feat_mean_apt_score":           "APT\n(mean)",
    "feat_max_dataleak_score":       "Data Leak\n(max)",
    "feat_mean_dataleak_score":      "Data Leak\n(mean)",
    "feat_max_dos_score":            "DoS\n(max)",
    "feat_mean_dos_score":           "DoS\n(mean)",

    # --- Resource ---
    "feat_vim0_cpu":                 "VIM0 CPU",
    "feat_vim0_ram":                 "VIM0 RAM",
    "feat_vim1_cpu":                 "VIM1 CPU",
    "feat_vim1_ram":                 "VIM1 RAM",
    "feat_min_remaining_mig":        "Migrate Budget\n(min)",
    "feat_min_remaining_reinst":     "Restart Budget\n(min)",
    "feat_mean_remaining_mig":       "Migrate Budget\n(mean)",
    "feat_mean_remaining_reinst":    "Restart Budget\n(mean)",

    # --- Network ---
    "feat_total_ues":                "Total UEs",
}

# Envelope's three reward objectives — used only by the Envelope-specific
# diagnostic plots (6, 7), which read the per-objective Φ_s summaries.
ENVELOPE_OBJECTIVES = ["resource", "network", "security"]

# Chosen-action Φ_s summary filename per algo (under data_root/<algo>/).
# All five entries are structurally equivalent: mean |SHAP| / mean SHAP for
# the action the policy actually selects (argmax Q / argmax prob / argmax
# scalarized Q). This is what the cross-algo comparison plots (4,5,8,9,11)
# read via _custom_summary.
ALGO_SUMMARY_FILE = {
    "dqn":      "shap_dqn_scalar_Q_summary.csv",
    "envelope": "shap_envelope_scalar_Q_summary.csv",
    "eupg":     "shap_eupg_policy_prob_summary.csv",
    "ppo":      "shap_ppo_policy_prob_summary.csv",
    "a2c":      "shap_a2c_policy_prob_summary.csv",
}

TARGET_LABELS = {
    "dqn":      "Q-value (scalar)",
    "envelope": "Q-value (scalarized)",
    "eupg":     "Policy probability π",
    "ppo":      "Policy probability π",
    "a2c":      "Policy probability π",
}


# ═══════════════════════════════════════════════════════════════════════════
class SHAPVisualizer:
    def __init__(self, data_root: str, out_dir: str = "figures", top_k: int = 5):
        self.root    = data_root
        self.out_dir = out_dir
        self.top_k   = top_k
        os.makedirs(out_dir, exist_ok=True)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _path(self, algo: str, fname: str) -> str:
        """Files are stored under data_root/<algo>/<fname>."""
        return os.path.join(self.root, algo, fname)

    def _load(self, algo: str, fname: str) -> pd.DataFrame | None:
        p = self._path(algo, fname)
        if not os.path.exists(p):
            print(f"  [SKIP] {p}")
            return None
        return pd.read_csv(p)

    def _save(self, fig: plt.Figure, name: str):
        path = os.path.join(self.out_dir, name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  ✓ saved → {path}")

    def _custom_summary(self, algo: str) -> pd.DataFrame | None:
        """
        Load the chosen-action Φ_s feature-importance summary for an algo.

        Generic for all five algorithms (DQN, Envelope, PPO, A2C, EUPG):
        each reads a single ALGO_SUMMARY_FILE[algo], which summarises Φ_s
        for the action the policy actually selects (mirrors _load_shap in
        silver_explain.py). For Envelope this is shap_envelope_scalar_Q_summary.csv
        (scalarized Q, argmax(scalar_q_a{i})) — the per-objective summaries
        (shap_envelope_Q_{resource,network,security}_summary.csv) are NOT
        used here; they feed the Envelope-only diagnostic plots (6, 7) instead.

        Returns DataFrame with columns [feature, mean_abs_shap, mean_shap],
        sorted by mean_abs_shap descending.
        """
        fname = ALGO_SUMMARY_FILE.get(algo)
        if fname is None:
            return None
        return self._load(algo, fname)

    # ══════════════════════════════════════════════════════════════════════
    # 4.  Signed mean SHAP bar chart per algo
    # ══════════════════════════════════════════════════════════════════════
    def plot_custom_signed_bars(self):
        """Signed mean SHAP bars: positive = pushes target up, negative = down."""
        print("\n[4] Customized SHAP — signed mean SHAP bars")
        fig, axes = plt.subplots(1, len(ALGOS), figsize=(18, 5), sharey=False)
        fig.suptitle(
            "Customized SHAP: Signed Mean SHAP per Algorithm\n"
            "(positive → increases target;  negative → decreases target)",
            fontsize=12, y=1.02,
        )

        for ax, algo in zip(axes, ALGOS):
            df = self._custom_summary(algo)
            if df is None:
                ax.set_visible(False)
                continue
            top    = df.head(self.top_k)
            labels = [SHORT_NAMES.get(f, f) for f in top["feature"]]
            vals   = top["mean_shap"].values
            colors = [ALGO_COLORS[algo] if v >= 0 else "#888888" for v in vals]
            ax.barh(range(len(top)), vals, color=colors, alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(algo.upper(), fontsize=11,
                         color=ALGO_COLORS[algo], fontweight="bold")
            ax.set_xlabel("mean SHAP", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)

        fig.tight_layout()
        self._save(fig, "04_custom_signed_bars.png")

    # ══════════════════════════════════════════════════════════════════════
    # 5.  Cross-algo radar chart
    # ══════════════════════════════════════════════════════════════════════
    def plot_custom_radar(self):
        """Radar chart comparing all 5 objective features across algos."""
        print("\n[5] Customized SHAP — cross-algo radar chart")

        feature_scores = {}
        for algo in ALGOS:
            df = self._custom_summary(algo)
            if df is not None:
                feature_scores[algo] = df.set_index("feature")["mean_abs_shap"]

        if len(feature_scores) < 2:
            print("  [SKIP] not enough algos")
            return

        # with only 5 features, use all of them
        radar_feats = FEATURE_COLS
        N_RADAR     = len(radar_feats)
        angles      = np.linspace(0, 2 * np.pi, N_RADAR, endpoint=False).tolist()
        angles     += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        for algo, s in feature_scores.items():
            vals      = [s.get(f, 0) for f in radar_feats]
            vmax      = max(vals) if max(vals) > 0 else 1
            vals_norm = [v / vmax for v in vals] + [vals[0] / vmax]
            ax.plot(angles, vals_norm, color=ALGO_COLORS[algo],
                    linewidth=2, label=algo.upper())
            ax.fill(angles, vals_norm, color=ALGO_COLORS[algo], alpha=0.08)

        labels = [SHORT_NAMES.get(f, f).replace("\n", " ") for f in radar_feats]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels([])
        ax.set_title("Customized SHAP: Feature Importance Radar\n"
                     "(normalised per algo)", fontsize=11, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        fig.tight_layout()
        self._save(fig, "05_custom_radar.png")

    # ══════════════════════════════════════════════════════════════════════
    # 6.  Envelope — objective influence pie + bar
    # ══════════════════════════════════════════════════════════════════════
    def plot_envelope_objective_influence(self):
        print("\n[6] Envelope — objective influence")
        df = self._load("envelope", "shap_envelope_objective_influence.csv")
        if df is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Envelope: Objective Influence on Action Selection", fontsize=13)
        colors = [OBJ_COLORS[o] for o in df["objective"]]

        # pie
        wedges, texts, autotexts = ax1.pie(
            df["influence_pct"],
            labels=df["objective"].str.capitalize(),
            colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax1.set_title("Influence %", fontsize=10)

        # bar
        bars = ax2.bar(
            df["objective"].str.capitalize(), df["mean_weighted_shap"],
            color=colors, edgecolor="white", width=0.5,
        )
        for bar, w in zip(bars, df["weight"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"w={w:.2f}", ha="center", fontsize=9,
            )
        ax2.set_ylabel("Mean Weighted |SHAP|")
        ax2.set_title("Weighted SHAP Magnitude", fontsize=10)

        fig.tight_layout()
        self._save(fig, "06_envelope_objective_influence.png")

    # ══════════════════════════════════════════════════════════════════════
    # 7.  Envelope — per-objective feature importance comparison
    # ══════════════════════════════════════════════════════════════════════
    def plot_envelope_per_objective(self):
        """Grouped bar: all 5 features for each of the 3 Envelope objectives."""
        print("\n[7] Envelope — per-objective feature importance")
        frames = {}
        for obj in ENVELOPE_OBJECTIVES:
            df = self._load("envelope", f"shap_envelope_Q_{obj}_summary.csv")
            if df is not None:
                frames[obj] = df.set_index("feature")["mean_abs_shap"]

        if not frames:
            return

        # use all 5 features in canonical order
        feats  = FEATURE_COLS
        matrix = pd.DataFrame(frames).reindex(feats).fillna(0)
        x      = np.arange(len(feats))
        width  = 0.26

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, obj in enumerate(ENVELOPE_OBJECTIVES):
            ax.bar(
                x + (i - 1) * width, matrix[obj], width,
                label=obj.capitalize(),
                color=OBJ_COLORS[obj], alpha=0.85, edgecolor="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHORT_NAMES.get(f, f).replace("\n", " ") for f in feats],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_ylabel("mean |SHAP|")
        ax.set_title(
            "Envelope: Feature Importance per Objective (Resource / Network / Security)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._save(fig, "07_envelope_per_objective.png")

    # ══════════════════════════════════════════════════════════════════════
    # 8.  Cross-algo — Spearman rank correlation matrix
    # ══════════════════════════════════════════════════════════════════════
    def plot_rank_correlation(self):
        print("\n[8] Cross-algo — Spearman rank correlation")
        importance = {}
        for algo in ALGOS:
            df = self._custom_summary(algo)
            if df is not None:
                s = df.set_index("feature")["mean_abs_shap"].reindex(FEATURE_COLS).fillna(0)
                importance[algo.upper()] = s.values

        algos_available = list(importance.keys())
        if len(algos_available) < 2:
            return

        n           = len(algos_available)
        corr_matrix = np.zeros((n, n))
        for i, a in enumerate(algos_available):
            for j, b in enumerate(algos_available):
                corr_matrix[i, j], _ = spearmanr(importance[a], importance[b])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Spearman ρ")
        ax.set_xticks(range(n))
        ax.set_xticklabels(algos_available, fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(algos_available, fontsize=10)
        ax.set_title(
            "Cross-Algorithm Feature Importance\nSpearman Rank Correlation",
            fontsize=11, pad=10,
        )
        for i in range(n):
            for j in range(n):
                v = corr_matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color="black" if abs(v) < 0.7 else "white")
        fig.tight_layout()
        self._save(fig, "08_crossalgo_spearman_corr.png")

    # ══════════════════════════════════════════════════════════════════════
    # 9.  Cross-algo — feature importance divergence heatmap
    # ══════════════════════════════════════════════════════════════════════
    def plot_importance_divergence(self):
        """Std dev of normalised importances across algos — high = disagreement."""
        print("\n[9] Cross-algo — feature importance divergence")
        data_dict = {}
        for algo in ALGOS:
            df = self._custom_summary(algo)
            if df is not None:
                s = df.set_index("feature")["mean_abs_shap"].reindex(FEATURE_COLS).fillna(0)
                data_dict[algo.upper()] = s / (s.max() or 1)

        if len(data_dict) < 2:
            return

        matrix   = pd.DataFrame(data_dict)   # (5 features, N algos)
        mean_imp = matrix.mean(axis=1)
        std_imp  = matrix.std(axis=1)
        order    = mean_imp.sort_values(ascending=False).index

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # left: mean importance per algo
        ax      = axes[0]
        mat_s   = matrix.loc[order]
        y       = np.arange(len(order))
        width   = 0.15
        for k, col in enumerate(matrix.columns):
            color = ALGO_COLORS.get(col.lower(), "#999999")
            ax.barh(y - k * width + width * 2, mat_s[col], width,
                    label=col, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [SHORT_NAMES.get(f, f).replace("\n", " ") for f in order], fontsize=9
        )
        ax.invert_yaxis()
        ax.set_xlabel("Normalised mean |SHAP|")
        ax.set_title("Feature Importance per Algorithm\n(normalised)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")

        # right: divergence (std)
        ax         = axes[1]
        std_sorted = std_imp.loc[order]
        colors_div = plt.cm.Reds(np.linspace(0.3, 0.9, len(order)))[::-1]
        ax.barh(y, std_sorted.values, color=colors_div, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [SHORT_NAMES.get(f, f).replace("\n", " ") for f in order], fontsize=9
        )
        ax.invert_yaxis()
        ax.set_xlabel("Std dev of normalised |SHAP|  (↑ = higher disagreement)")
        ax.set_title("Cross-Algorithm Disagreement\nper Feature", fontsize=10)

        fig.suptitle(
            "Feature Importance: Distribution & Divergence Across Algorithms",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        self._save(fig, "09_crossalgo_divergence.png")

    # ══════════════════════════════════════════════════════════════════════
    # 11. Dashboard — top-k features per algo side-by-side
    # ══════════════════════════════════════════════════════════════════════
    def plot_dashboard_summary(self):
        print("\n[11] Dashboard summary")
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("#F8F9FA")
        fig.suptitle(
            "SHAP Analysis Dashboard — Algorithm Comparison\n"
            f"Top {self.top_k} Features by Customized SHAP Importance",
            fontsize=15, fontweight="bold", y=0.98,
        )

        gs        = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        for (r, c), algo in zip(positions, ALGOS):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("white")
            df = self._custom_summary(algo)
            if df is None:
                ax.axis("off")
                continue
            top    = df.head(self.top_k)
            labels = [SHORT_NAMES.get(f, f).replace("\n", " ") for f in top["feature"]]
            vals   = top["mean_abs_shap"].values
            color  = ALGO_COLORS[algo]
            bars   = ax.barh(range(len(top)), vals, color=color,
                             alpha=0.9, edgecolor="white", height=0.6)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("mean |SHAP|", fontsize=7)
            ax.set_title(
                f"{algo.upper()}\n↳ {TARGET_LABELS[algo]}",
                fontsize=9, color=color, fontweight="bold", pad=4,
            )
            for bar, v in zip(bars, vals):
                ax.text(v, bar.get_y() + bar.get_height() / 2,
                        f" {v:.3f}", va="center", fontsize=6.5)
            ax.tick_params(labelsize=6.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # blank last cell → legend
        ax_leg = fig.add_subplot(gs[1, 2])
        ax_leg.axis("off")
        patches = [mpatches.Patch(color=ALGO_COLORS[a], label=a.upper()) for a in ALGOS]
        ax_leg.legend(handles=patches, loc="center", fontsize=11,
                      title="Algorithms", title_fontsize=10, frameon=False)

        self._save(fig, "11_dashboard_summary.png")

    # ══════════════════════════════════════════════════════════════════════
    # Entry point
    # ══════════════════════════════════════════════════════════════════════
    def run_all(self):
        print(f"\n{'='*60}")
        print(f"  SHAP Visualization — data root: {self.root}")
        print(f"  Output dir: {self.out_dir}")
        print(f"{'='*60}")
        self.plot_custom_signed_bars()           # 4
        self.plot_custom_radar()                 # 5
        self.plot_envelope_objective_influence() # 6
        self.plot_envelope_per_objective()       # 7
        self.plot_rank_correlation()             # 8
        self.plot_importance_divergence()        # 9
        self.plot_dashboard_summary()            # 11
        print(f"\n✓ All plots saved to: {self.out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP output visualization & cross-algorithm comparison"
    )
    parser.add_argument("--data_root", required=True,
                        help="Folder containing shap_*.csv files (output of shap_explain.py)")
    parser.add_argument("--out_dir", default="shap_figures",
                        help="Directory where PNG figures are saved (default: shap_figures/)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top features to display (default: 5, max: 5)")
    parser.add_argument("--plot", default="all",
                        help="Comma-separated plot numbers to run, or 'all' "
                             "(choices: 4,5,6,7,8,9,11). Default: all")
    args = parser.parse_args()

    args.top_k = min(args.top_k, len(FEATURE_COLS))  # cap at 5

    viz = SHAPVisualizer(args.data_root, args.out_dir, args.top_k)

    if args.plot == "all":
        viz.run_all()
    else:
        plot_map = {
            "4":  viz.plot_custom_signed_bars,
            "5":  viz.plot_custom_radar,
            "6":  viz.plot_envelope_objective_influence,
            "7":  viz.plot_envelope_per_objective,
            "8":  viz.plot_rank_correlation,
            "9":  viz.plot_importance_divergence,
            "11": viz.plot_dashboard_summary,
        }
        for num in args.plot.split(","):
            num = num.strip()
            if num in plot_map:
                plot_map[num]()
            else:
                print(f"[WARNING] Unknown plot number: {num}")