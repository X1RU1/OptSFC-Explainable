"""
SHAP Analysis Visualization & Comparison
=========================================
Reads the SHAP output CSVs produced by shap_explain.py and generates:

  1.  Universal SHAP — per-algo feature importance bar chart (top-K)
  2.  Universal SHAP — cross-algo heatmap (feature rank correlation)
  3.  Universal SHAP — per-action importance heatmap (action × feature)
  4.  Customized SHAP — per-algo summary bar chart (signed mean SHAP)
  5.  Customized SHAP — cross-algo scatter / radar (top features)
  6.  Envelope only  — objective influence pie + bar
  7.  Envelope only  — per-objective feature importance comparison
  8.  Cross-algo     — rank-based feature agreement matrix (Spearman)
  9.  Cross-algo     — feature importance divergence heatmap

Usage
-----
  python shap_analysis_viz.py --data_root ./shap_outputs --out_dir ./figures

  # Or import and call directly:
  from shap_analysis_viz import SHAPVisualizer
  viz = SHAPVisualizer("./shap_outputs")
  viz.run_all()
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
N_ACTIONS = 12
FEATURE_COLS = [
    "feat_vim0_cpu", "feat_vim0_ram", "feat_vim1_cpu", "feat_vim1_ram",
    "feat_max_apt_score", "feat_mean_apt_score",
    "feat_max_dataleak_score", "feat_mean_dataleak_score",
    "feat_max_dos_score", "feat_mean_dos_score",
    "feat_mean_security_penalty", "feat_max_security_penalty",
    "feat_security_penalty_cumul",
    "feat_mean_network_penalty", "feat_max_network_penalty",
    "feat_mean_mtd_overhead",
    "feat_min_remaining_mig", "feat_mean_remaining_mig",
    "feat_min_remaining_reinst", "feat_mean_remaining_reinst",
    # "feat_steps_since_last_mtd",
    "feat_total_ues", "feat_nb_resources",
]
SHORT_NAMES = {f: f.replace("feat_", "").replace("_", "\n") for f in FEATURE_COLS}

ENVELOPE_OBJECTIVES = ["resource", "network", "security"]


# ═══════════════════════════════════════════════════════════════════════════
class SHAPVisualizer:
    def __init__(self, data_root: str, out_dir: str = "figures", top_k: int = 12):
        self.root = data_root
        self.out_dir = out_dir
        self.top_k = top_k
        os.makedirs(out_dir, exist_ok=True)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _path(self, algo: str, fname: str) -> str:
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

    def _universal_summary(self, algo: str) -> pd.DataFrame | None:
        """Merge per-action universal summaries → mean |SHAP| per feature."""
        frames = []
        for a in range(N_ACTIONS):
            df = self._load(algo, f"shap_universal_summary_action{a}.csv")
            if df is not None:
                frames.append(df.set_index("feature")["mean_abs_shap"])
        if not frames:
            return None
        merged = pd.concat(frames, axis=1).mean(axis=1).reset_index()
        merged.columns = ["feature", "mean_abs_shap"]
        return merged.sort_values("mean_abs_shap", ascending=False)

    def _custom_summary(self, algo: str) -> pd.DataFrame | None:
        """Load customized SHAP summary for an algo."""
        mapping = {
            "dqn":      "shap_dqn_scalar_Q_summary.csv",
            "eupg":     "shap_eupg_policy_prob_summary.csv",
            "ppo":      "shap_ppo_policy_prob_summary.csv",
            "a2c":      "shap_a2c_policy_prob_summary.csv",
        }
        if algo == "envelope":
            # average the 3 objective summaries
            frames = []
            for obj in ENVELOPE_OBJECTIVES:
                df = self._load(algo, f"shap_envelope_Q_{obj}_summary.csv")
                if df is not None:
                    frames.append(df.set_index("feature"))
            if not frames:
                return None
            merged = pd.concat(frames).groupby(level=0).mean().reset_index()
            merged = merged.sort_values("mean_abs_shap", ascending=False)
            return merged
        fname = mapping.get(algo)
        return self._load(algo, fname) if fname else None

    # ══════════════════════════════════════════════════════════════════════
    # 1.  Universal SHAP — per-algo feature importance bar chart
    # ══════════════════════════════════════════════════════════════════════
    def plot_universal_per_algo(self):
        """One bar chart per algo showing top-K features by mean |SHAP|."""
        print("\n[1] Universal SHAP — per-algo importance bars")
        fig, axes = plt.subplots(1, len(ALGOS), figsize=(22, 6), sharey=False)
        fig.suptitle("Universal SHAP: Top Feature Importance per Algorithm\n"
                     "(mean |SHAP| averaged over all 12 actions)", fontsize=13, y=1.02)

        for ax, algo in zip(axes, ALGOS):
            df = self._universal_summary(algo)
            if df is None:
                ax.set_visible(False)
                continue
            top = df.head(self.top_k)
            labels = [SHORT_NAMES.get(f, f) for f in top["feature"]]
            vals   = top["mean_abs_shap"].values
            color  = ALGO_COLORS[algo]
            bars   = ax.barh(range(len(top)), vals, color=color, alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(algo.upper(), fontsize=11, color=color, fontweight="bold")
            ax.set_xlabel("mean |SHAP|", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            # value labels
            for bar, v in zip(bars, vals):
                ax.text(v * 1.02, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=6)

        fig.tight_layout()
        self._save(fig, "01_universal_per_algo_importance.png")

    # ══════════════════════════════════════════════════════════════════════
    # 2.  Universal SHAP — cross-algo feature importance heatmap
    # ══════════════════════════════════════════════════════════════════════
    def plot_universal_crossalgo_heatmap(self):
        """Heatmap: features × algos, cell = mean |SHAP| (normalised per algo)."""
        print("\n[2] Universal SHAP — cross-algo heatmap")
        data_dict = {}
        for algo in ALGOS:
            df = self._universal_summary(algo)
            if df is not None:
                s = df.set_index("feature")["mean_abs_shap"]
                data_dict[algo.upper()] = s / s.max()   # normalise 0-1 per algo

        if not data_dict:
            return
        matrix = pd.DataFrame(data_dict).reindex(FEATURE_COLS).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Normalised mean |SHAP|")

        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, fontsize=10)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([SHORT_NAMES.get(f, f).replace("\n", " ")
                            for f in matrix.index], fontsize=7.5)
        ax.set_title("Universal SHAP: Normalised Feature Importance Across Algorithms",
                     fontsize=12, pad=12)

        # annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if v < 0.6 else "white")

        fig.tight_layout()
        self._save(fig, "02_universal_crossalgo_heatmap.png")

    # ══════════════════════════════════════════════════════════════════════
    # 3.  Universal SHAP — per-action importance heatmap (one algo)
    # ══════════════════════════════════════════════════════════════════════
    def plot_universal_action_heatmap(self):
        """For each algo: actions × features heatmap of mean |SHAP|."""
        print("\n[3] Universal SHAP — per-action × feature heatmap")
        for algo in ALGOS:
            rows = {}
            for a in range(N_ACTIONS):
                df = self._load(algo, f"shap_universal_summary_action{a}.csv")
                if df is not None:
                    rows[a] = df.set_index("feature")["mean_abs_shap"]
            if not rows:
                continue
            matrix = pd.DataFrame(rows).T.reindex(columns=FEATURE_COLS).fillna(0)
            # normalise per action
            matrix_norm = matrix.div(matrix.max(axis=1).replace(0, 1), axis=0)

            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(matrix_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label="Normalised mean |SHAP|")
            ax.set_yticks(range(N_ACTIONS))
            ax.set_yticklabels([f"Action {i}" for i in range(N_ACTIONS)], fontsize=8)
            ax.set_xticks(range(len(FEATURE_COLS)))
            ax.set_xticklabels([SHORT_NAMES[f].replace("\n", "\n") for f in FEATURE_COLS],
                               fontsize=6, rotation=45, ha="right")
            ax.set_title(f"{algo.upper()} — Universal SHAP: Action × Feature Importance",
                         fontsize=11, color=ALGO_COLORS[algo], pad=10)
            fig.tight_layout()
            self._save(fig, f"03_universal_action_heatmap_{algo}.png")

    # ══════════════════════════════════════════════════════════════════════
    # 4.  Customized SHAP — signed mean SHAP bar chart per algo
    # ══════════════════════════════════════════════════════════════════════
    def plot_custom_signed_bars(self):
        """Signed mean SHAP bars: positive = pushes target up, negative = down."""
        print("\n[4] Customized SHAP — signed mean SHAP bars")
        fig, axes = plt.subplots(1, len(ALGOS), figsize=(22, 6), sharey=False)
        fig.suptitle("Customized SHAP: Signed Mean SHAP per Algorithm\n"
                     "(positive → increases target; negative → decreases target)",
                     fontsize=13, y=1.02)

        for ax, algo in zip(axes, ALGOS):
            df = self._custom_summary(algo)
            if df is None:
                ax.set_visible(False)
                continue
            top = df.head(self.top_k)
            labels = [SHORT_NAMES.get(f, f) for f in top["feature"]]
            vals   = top["mean_shap"].values
            colors = [ALGO_COLORS[algo] if v >= 0 else "#888888" for v in vals]
            bars   = ax.barh(range(len(top)), vals, color=colors, alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(algo.upper(), fontsize=11, color=ALGO_COLORS[algo], fontweight="bold")
            ax.set_xlabel("mean SHAP", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)

        fig.tight_layout()
        self._save(fig, "04_custom_signed_bars.png")

    # ══════════════════════════════════════════════════════════════════════
    # 5.  Customized SHAP — cross-algo radar / spider chart (top shared features)
    # ══════════════════════════════════════════════════════════════════════
    def plot_custom_radar(self):
        """Radar chart comparing top-N shared features across all algos."""
        print("\n[5] Customized SHAP — cross-algo radar chart")
        N_RADAR = 8

        # collect top features per algo
        feature_scores = {}
        for algo in ALGOS:
            df = self._custom_summary(algo)
            if df is not None:
                feature_scores[algo] = df.set_index("feature")["mean_abs_shap"]

        if len(feature_scores) < 2:
            print("  [SKIP] not enough algos")
            return

        # select features by average rank
        all_feat = set()
        for s in feature_scores.values():
            all_feat |= set(s.index)
        rank_df = pd.DataFrame({
            algo: s.rank(ascending=False) for algo, s in feature_scores.items()
        }).reindex(list(all_feat)).fillna(999)
        top_feats = rank_df.mean(axis=1).nsmallest(N_RADAR).index.tolist()

        # build radar
        angles = np.linspace(0, 2 * np.pi, N_RADAR, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        for algo, s in feature_scores.items():
            vals = [s.get(f, 0) for f in top_feats]
            vmax = max(vals) if max(vals) > 0 else 1
            vals_norm = [v / vmax for v in vals] + [vals[0] / vmax]
            ax.plot(angles, vals_norm, color=ALGO_COLORS[algo], linewidth=2,
                    label=algo.upper())
            ax.fill(angles, vals_norm, color=ALGO_COLORS[algo], alpha=0.08)

        labels = [SHORT_NAMES.get(f, f).replace("\n", " ") for f in top_feats]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title("Customized SHAP: Top Feature Importance Radar\n"
                     "(normalised per algo)", fontsize=11, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        fig.tight_layout()
        self._save(fig, "05_custom_radar.png")

    # ══════════════════════════════════════════════════════════════════════
    # 6.  Envelope — objective influence
    # ══════════════════════════════════════════════════════════════════════
    def plot_envelope_objective_influence(self):
        """Pie + bar for Envelope objective influence."""
        print("\n[6] Envelope — objective influence")
        df = self._load("envelope", "shap_envelope_objective_influence.csv")
        if df is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Envelope: Objective Influence on Action Selection", fontsize=13)

        colors = [OBJ_COLORS[o] for o in df["objective"]]

        # pie
        wedges, texts, autotexts = ax1.pie(
            df["influence_pct"], labels=df["objective"].str.capitalize(),
            colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax1.set_title("Influence %", fontsize=10)

        # bar
        bars = ax2.bar(df["objective"].str.capitalize(), df["mean_weighted_shap"],
                       color=colors, edgecolor="white", width=0.5)
        for bar, w in zip(bars, df["weight"]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.02,
                     f"w={w:.2f}", ha="center", fontsize=9)
        ax2.set_ylabel("Mean Weighted |SHAP|")
        ax2.set_title("Weighted SHAP Magnitude", fontsize=10)

        fig.tight_layout()
        self._save(fig, "06_envelope_objective_influence.png")

    # ══════════════════════════════════════════════════════════════════════
    # 7.  Envelope — per-objective feature importance comparison
    # ══════════════════════════════════════════════════════════════════════
    def plot_envelope_per_objective(self):
        """Grouped bar: top features for each of the 3 Envelope objectives."""
        print("\n[7] Envelope — per-objective feature importance")
        frames = {}
        for obj in ENVELOPE_OBJECTIVES:
            df = self._load("envelope", f"shap_envelope_Q_{obj}_summary.csv")
            if df is not None:
                frames[obj] = df.set_index("feature")["mean_abs_shap"]

        if not frames:
            return

        # union top features
        top_feats = []
        for s in frames.values():
            top_feats += s.nlargest(self.top_k).index.tolist()
        # rank by average importance across objectives
        avg = pd.DataFrame(frames).reindex(list(set(top_feats))).fillna(0).mean(axis=1)
        top_feats = avg.nlargest(self.top_k).index.tolist()

        matrix = pd.DataFrame(frames).reindex(top_feats).fillna(0)
        x = np.arange(len(top_feats))
        width = 0.26

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, obj in enumerate(ENVELOPE_OBJECTIVES):
            ax.bar(x + i * width - width, matrix[obj], width,
                   label=obj.capitalize(), color=OBJ_COLORS[obj], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_NAMES.get(f, f).replace("\n", " ") for f in top_feats],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mean |SHAP|")
        ax.set_title("Envelope: Feature Importance per Objective (Resource / Network / Security)",
                     fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._save(fig, "07_envelope_per_objective.png")

    # ══════════════════════════════════════════════════════════════════════
    # 8.  Cross-algo — Spearman rank correlation matrix
    # ══════════════════════════════════════════════════════════════════════
    def plot_rank_correlation(self):
        """Spearman correlation of feature importance ranks across algos."""
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

        n = len(algos_available)
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
        ax.set_title("Cross-Algorithm Feature Importance\nSpearman Rank Correlation",
                     fontsize=11, pad=10)
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
        """
        How much does each algo *disagree* about each feature?
        Cell = std dev of normalised importances across algos.
        High std → high disagreement.
        """
        print("\n[9] Cross-algo — feature importance divergence")
        data_dict = {}
        for algo in ALGOS:
            df = self._custom_summary(algo)
            if df is not None:
                s = df.set_index("feature")["mean_abs_shap"].reindex(FEATURE_COLS).fillna(0)
                data_dict[algo.upper()] = s / (s.max() or 1)

        if len(data_dict) < 2:
            return

        matrix = pd.DataFrame(data_dict)          # (22 features, N algos)
        mean_imp = matrix.mean(axis=1)
        std_imp  = matrix.std(axis=1)

        # sort by mean importance descending
        order = mean_imp.sort_values(ascending=False).index

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # left: mean importance per algo (stacked-style grouped bars)
        ax = axes[0]
        mat_sorted = matrix.loc[order]
        y = np.arange(len(order))
        width = 0.15
        for k, (col, color) in enumerate(zip(matrix.columns,
                                             [ALGO_COLORS[a.lower()]
                                              for a in matrix.columns])):
            ax.barh(y - k * width + width * 2, mat_sorted[col],
                    width, label=col, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT_NAMES.get(f, f).replace("\n", " ") for f in order],
                           fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised mean |SHAP|")
        ax.set_title("Feature Importance per Algorithm\n(normalised)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

        # right: divergence (std)
        ax = axes[1]
        std_sorted = std_imp.loc[order]
        colors_div = plt.cm.Reds(np.linspace(0.3, 0.9, len(order)))[::-1]
        bars = ax.barh(y, std_sorted.values, color=colors_div, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT_NAMES.get(f, f).replace("\n", " ") for f in order],
                           fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Std dev of normalised |SHAP| (↑ = higher disagreement)")
        ax.set_title("Cross-Algorithm Disagreement\nper Feature", fontsize=10)

        fig.suptitle("Feature Importance: Distribution & Divergence Across Algorithms",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        self._save(fig, "09_crossalgo_divergence.png")

    # ══════════════════════════════════════════════════════════════════════
    # 10.  BONUS — Universal vs Customized agreement per algo
    # ══════════════════════════════════════════════════════════════════════
    def plot_universal_vs_custom(self):
        """
        For each algo, compare the feature ranking from Universal SHAP vs
        Customized SHAP using a scatter plot (rank in universal vs rank in custom).
        """
        print("\n[10] Universal vs Customized SHAP agreement per algo")
        fig, axes = plt.subplots(1, len(ALGOS), figsize=(22, 5))
        fig.suptitle("Universal vs Customized SHAP: Feature Rank Agreement per Algorithm",
                     fontsize=13, y=1.02)

        for ax, algo in zip(axes, ALGOS):
            univ_df = self._universal_summary(algo)
            cust_df = self._custom_summary(algo)
            if univ_df is None or cust_df is None:
                ax.set_visible(False)
                continue

            univ_ranks = univ_df.set_index("feature")["mean_abs_shap"].rank(ascending=False)
            cust_ranks = cust_df.set_index("feature")["mean_abs_shap"].rank(ascending=False)

            common = univ_ranks.index.intersection(cust_ranks.index)
            x = univ_ranks[common].values
            y = cust_ranks[common].values

            corr, _ = spearmanr(x, y)
            ax.scatter(x, y, color=ALGO_COLORS[algo], alpha=0.7, s=60, edgecolors="white")
            # label top features
            for feat in common:
                if cust_ranks[feat] <= 5 or univ_ranks[feat] <= 5:
                    ax.annotate(SHORT_NAMES.get(feat, feat).replace("\n", " "),
                                (univ_ranks[feat], cust_ranks[feat]),
                                fontsize=5.5, ha="center",
                                xytext=(2, 2), textcoords="offset points")

            # diagonal
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([1, lim], [1, lim], "k--", linewidth=0.7, alpha=0.4)
            ax.set_xlabel("Universal rank", fontsize=8)
            ax.set_ylabel("Customized rank", fontsize=8)
            ax.set_title(f"{algo.upper()}\nSpearman ρ={corr:.2f}",
                         fontsize=10, color=ALGO_COLORS[algo], fontweight="bold")
            ax.tick_params(labelsize=7)

        fig.tight_layout()
        self._save(fig, "10_universal_vs_custom_rank.png")

    # ══════════════════════════════════════════════════════════════════════
    # 11. Summary dashboard: top-5 features per algo side-by-side
    # ══════════════════════════════════════════════════════════════════════
    def plot_dashboard_summary(self):
        """One-page summary: top-5 features (custom SHAP) for each algo."""
        print("\n[11] Dashboard summary")
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("#F8F9FA")
        fig.suptitle("SHAP Analysis Dashboard — Algorithm Comparison\n"
                     "Top 5 Features by Customized SHAP Importance",
                     fontsize=15, fontweight="bold", y=0.98)

        gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        target_labels = {
            "dqn": "Q-value (scalar)",
            "envelope": "Q-value (averaged objectives)",
            "eupg": "Policy probability π",
            "ppo": "Policy probability π",
            "a2c": "Policy probability π",
        }

        for (r, c), algo in zip(positions, ALGOS):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("white")
            df = self._custom_summary(algo)
            if df is None:
                ax.axis("off")
                continue
            top5 = df.head(5)
            labels = [SHORT_NAMES.get(f, f).replace("\n", " ") for f in top5["feature"]]
            vals   = top5["mean_abs_shap"].values
            color  = ALGO_COLORS[algo]
            bars   = ax.barh(range(5), vals, color=color, alpha=0.9, edgecolor="white",
                             height=0.6)
            ax.set_yticks(range(5))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("mean |SHAP|", fontsize=7)
            ax.set_title(f"{algo.upper()}\n↳ {target_labels[algo]}",
                         fontsize=9, color=color, fontweight="bold", pad=4)
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

        self.plot_universal_per_algo()           # 1
        self.plot_universal_crossalgo_heatmap()  # 2
        self.plot_universal_action_heatmap()     # 3
        self.plot_custom_signed_bars()           # 4
        self.plot_custom_radar()                 # 5
        self.plot_envelope_objective_influence() # 6
        self.plot_envelope_per_objective()       # 7
        self.plot_rank_correlation()             # 8
        self.plot_importance_divergence()        # 9
        self.plot_universal_vs_custom()          # 10
        self.plot_dashboard_summary()            # 11

        print(f"\n✓ All plots saved to: {self.out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP output visualization & cross-algorithm comparison"
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Root folder containing per-algo sub-folders (dqn/, envelope/, ...)"
    )
    parser.add_argument(
        "--out_dir", default="shap_figures",
        help="Directory where PNG figures are saved (default: shap_figures/)"
    )
    parser.add_argument(
        "--top_k", type=int, default=12,
        help="Number of top features to display (default: 12)"
    )
    parser.add_argument(
        "--plot", default="all",
        help=(
            "Which plot(s) to generate. Use 'all' or comma-separated numbers, "
            "e.g. '1,4,8'  (1=universal bars, 2=crossalgo heatmap, "
            "3=action heatmap, 4=signed bars, 5=radar, 6=envelope influence, "
            "7=envelope objectives, 8=spearman, 9=divergence, "
            "10=univ vs custom, 11=dashboard)"
        )
    )
    args = parser.parse_args()

    viz = SHAPVisualizer(args.data_root, args.out_dir, args.top_k)

    if args.plot == "all":
        viz.run_all()
    else:
        plot_map = {
            "1":  viz.plot_universal_per_algo,
            "2":  viz.plot_universal_crossalgo_heatmap,
            "3":  viz.plot_universal_action_heatmap,
            "4":  viz.plot_custom_signed_bars,
            "5":  viz.plot_custom_radar,
            "6":  viz.plot_envelope_objective_influence,
            "7":  viz.plot_envelope_per_objective,
            "8":  viz.plot_rank_correlation,
            "9":  viz.plot_importance_divergence,
            "10": viz.plot_universal_vs_custom,
            "11": viz.plot_dashboard_summary,
        }
        for num in args.plot.split(","):
            num = num.strip()
            if num in plot_map:
                plot_map[num]()
            else:
                print(f"[WARNING] Unknown plot number: {num}")