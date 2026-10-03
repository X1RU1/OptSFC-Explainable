"""
Cross-Seed Metric Summary & Representative Seed Selection
============================================================
For each of the 5 RL algorithms (DQN, PPO, A2C, Envelope, EUPG), this script:
  1. Loads eval_explain.csv (or train_explain.csv, see USE_TRAIN_EXPLAIN below)
     for every seed found under ./trained_models/{model_name}_seed{seed}/
  2. Computes per-seed diagnostic metrics:
       - match_rate                : mean of the 'match' column (RDX agreement)
       - action_entropy            : Shannon entropy of env_action distribution
                                      (0 = fully collapsed to one action,
                                       higher = more diverse action usage)
       - n_unique_actions          : how many distinct actions were executed
       - dominant_action           : the most frequently executed action
       - dominant_action_share     : fraction of steps using dominant_action
       - mean_delta / mean_weighted_diffs : algorithm-specific Q-diff summary
  3. Prints a per-algorithm table across all seeds
  4. Selects a "representative" seed per algorithm — the one whose key
     metric (match_rate for deterministic algos, action_entropy for
     stochastic / MORL algos) is closest to the cross-seed median,
     after removing IQR outliers so a single anomalous seed doesn't skew
     the median.
  5. Saves one combined CSV (all_seeds_metrics.csv) and prints the final
     recommendation table.

Usage:
  python select_representative_seed.py
  (edit MODEL_NAMES / SEEDS / BASE_DIR / USE_TRAIN_EXPLAIN below as needed)
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = "./trained_models"
SEEDS      = [0, 1, 2, 3, 4]

# model_name -> (folder prefix, algo category)
# category is used to decide which metric drives seed selection:
#   "deterministic" -> match_rate  (DQN, Envelope)
#   "stochastic"    -> action_entropy (PPO, A2C, EUPG)
MODEL_CONFIG = {
    "dqn_model":      "deterministic",
    "ppo_model":      "stochastic",
    "a2c_model":      "stochastic",
    "envelope_model": "deterministic",
    "eupg_model":     "stochastic",
}

# Set True to analyze train_explain.csv instead of eval_explain.csv.
USE_TRAIN_EXPLAIN = True

MORL_DIFF_COLS = [
    "weighted_resource_diff",
    "weighted_network_diff",
    "weighted_security_diff",
]

OUT_CSV = "./all_seeds_metrics.csv"


# ── Per-seed metric computation ───────────────────────────────────────────────

def compute_seed_metrics(model_name: str, seed: int) -> dict | None:
    filename = "train_explain.csv" if USE_TRAIN_EXPLAIN else "eval_explain.csv"
    path = os.path.join(BASE_DIR, f"{model_name}_seed{seed}", filename)
    if not os.path.isfile(path):
        print(f"  [skip] {model_name} seed={seed}: file not found ({path})")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    metrics = {
        "model_name": model_name,
        "seed": seed,
        "n_rows": len(df),
    }

    # ── RDX match rate ────────────────────────────────────────────────────
    if "match" in df.columns:
        metrics["match_rate"] = float(df["match"].mean())
    else:
        metrics["match_rate"] = np.nan

    # ── Action distribution / collapse diagnostics ───────────────────────
    if "env_action" in df.columns and len(df) > 0:
        action_counts = df["env_action"].value_counts(normalize=True)
        entropy = float(-(action_counts * np.log(action_counts + 1e-12)).sum())
        metrics["action_entropy"]        = entropy
        metrics["n_unique_actions"]      = int(df["env_action"].nunique())
        metrics["dominant_action"]       = int(action_counts.idxmax())
        metrics["dominant_action_share"] = float(action_counts.max())
    else:
        metrics["action_entropy"]        = np.nan
        metrics["n_unique_actions"]      = np.nan
        metrics["dominant_action"]       = np.nan
        metrics["dominant_action_share"] = np.nan

    # ── Scalar delta summary (DQN / PPO / A2C) ───────────────────────────
    if "delta" in df.columns:
        metrics["mean_delta"] = float(df["delta"].mean())
        metrics["std_delta"]  = float(df["delta"].std())

    # ── MORL weighted diff summary (Envelope / EUPG) ─────────────────────
    available_diff_cols = [c for c in MORL_DIFF_COLS if c in df.columns]
    if available_diff_cols:
        for col in available_diff_cols:
            short_name = col.replace("weighted_", "").replace("_diff", "")
            metrics[f"mean_{short_name}_diff"] = float(df[col].mean())
        metrics["mean_total_abs_weighted_diff"] = float(
            df[available_diff_cols].abs().sum(axis=1).mean()
        )

    # ── Policy entropy, if logged (PPO / A2C / EUPG) ─────────────────────
    if "policy_entropy" in df.columns:
        metrics["mean_policy_entropy"] = float(df["policy_entropy"].mean())

    return metrics


# ── Representative seed selection (outlier-robust median matching) ──────────

def select_representative_seed(summary: pd.DataFrame, key_metric: str) -> int | None:
    """
    Select the seed whose `key_metric` is closest to the cross-seed median,
    after removing IQR outliers so a single anomalous seed does not distort
    the median. Falls back to plain median-matching if too few seeds remain
    after outlier removal (need at least 3 to compute IQR meaningfully).
    """
    valid = summary.dropna(subset=[key_metric])
    if valid.empty:
        return None

    if len(valid) >= 4:
        q1, q3 = valid[key_metric].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = valid[(valid[key_metric] >= lower) & (valid[key_metric] <= upper)]
        if not filtered.empty:
            valid = filtered

    median_val = valid[key_metric].median()
    dist = (valid[key_metric] - median_val).abs()
    return int(valid.loc[dist.idxmin(), "seed"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []
    recommendations = []

    for model_name, category in MODEL_CONFIG.items():
        print(f"\n{'='*70}")
        print(f"{model_name}  (category: {category})")
        print(f"{'='*70}")

        rows = []
        for seed in SEEDS:
            m = compute_seed_metrics(model_name, seed)
            if m is not None:
                rows.append(m)

        if not rows:
            print("  No data found for this algorithm — skipping.")
            continue

        summary = pd.DataFrame(rows).set_index("seed", drop=False)
        all_rows.extend(rows)

        display_cols = [c for c in [
            "n_rows", "match_rate", "action_entropy", "n_unique_actions",
            "dominant_action", "dominant_action_share",
            "mean_delta", "mean_total_abs_weighted_diff", "mean_policy_entropy",
        ] if c in summary.columns]

        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", None)
        print(summary[display_cols].to_string())

        # Choose the driving metric based on algorithm category.
        key_metric = "match_rate" if category == "deterministic" else "action_entropy"
        if key_metric not in summary.columns or summary[key_metric].isna().all():
            # fall back if the preferred metric is unavailable
            key_metric = "action_entropy" if key_metric == "match_rate" else "match_rate"

        selected_seed = select_representative_seed(summary, key_metric)
        median_val = summary[key_metric].median()

        print(f"\n  → Driving metric: {key_metric}")
        print(f"  → Cross-seed median {key_metric}: {median_val:.4f}")
        if selected_seed is not None:
            sel_val = summary.loc[selected_seed, key_metric]
            print(f"  → Recommended representative seed: {selected_seed} "
                  f"({key_metric}={sel_val:.4f})")
        else:
            print("  → Could not determine a representative seed (missing data).")

        recommendations.append({
            "model_name": model_name,
            "category": category,
            "key_metric": key_metric,
            "median_value": median_val,
            "recommended_seed": selected_seed,
        })

    # ── Save combined metrics table ───────────────────────────────────────
    if all_rows:
        combined = pd.DataFrame(all_rows)
        combined.to_csv(OUT_CSV, index=False)
        print(f"\nAll per-seed metrics saved to: {OUT_CSV}")

    # ── Final recommendation table ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION SUMMARY")
    print(f"{'='*70}")
    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        print(rec_df.to_string(index=False))
    else:
        print("No recommendations could be produced — check that eval_explain.csv "
              "files exist under ./trained_models/{model}_seed{n}/")


if __name__ == "__main__":
    main()