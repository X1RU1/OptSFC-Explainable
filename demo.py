# import pandas as pd
# import numpy as np

# REWARDS_COEFF = [0.4, 0.3, 0.3]  # [resource, network, security]
# GAMMA = 0.99
# HORIZON = 100  # 1 / (1 - gamma)

# def compute_horizon_return(df, gamma=GAMMA, horizon=HORIZON, coeff=REWARDS_COEFF):
#     """
#     Rolling discounted return over a fixed horizon, used as a reward-scale
#     reference to contextualize Q-value magnitudes (DQN scalar Q / Envelope
#     scalarized Q) reported in the SHAP analysis.

#     Not an episode return: budget_reset='daily' means this environment has
#     no natural episode boundary within the logged trajectory, so we use a
#     rolling window matching the algorithm's effective horizon instead.

#     Scalarizes reward_resource/reward_network/reward_security per step using
#     the same rewards_coeff weights used to train the model and compute
#     scalar_q_a{i}, so the resulting number is on the same scale as the
#     Q-value SHAP target.
#     """
#     missing = [c for c in ("reward_resource", "reward_network", "reward_security")
#                if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing reward columns: {missing}")

#     scalar_reward = (
#         df["reward_resource"].values * coeff[0]
#         + df["reward_network"].values * coeff[1]
#         + df["reward_security"].values * coeff[2]
#     )

#     discounts = gamma ** np.arange(horizon)
#     horizon_returns = np.convolve(scalar_reward, discounts[::-1], mode="valid")
#     return horizon_returns


# def report_reward_scale(csv_path, algo_label):
#     df = pd.read_csv(csv_path)
#     hr = compute_horizon_return(df)
#     print(f"[{algo_label}] n_windows={len(hr)}")
#     print(f"  mean:   {hr.mean():.4f}")
#     print(f"  median: {np.median(hr):.4f}")
#     print(f"  std:    {hr.std():.4f}")
#     print(f"  min/max: {hr.min():.4f} / {hr.max():.4f}")
#     return hr


# if __name__ == "__main__":
#     report_reward_scale("trained_models/dqn_model_seed0/train_explain.csv", "DQN")
#     report_reward_scale("trained_models/envelope_model_seed3/train_explain.csv", "Envelope")

# """
# Compute mean per-step reward components (Security / Network / Resource)
# for each algorithm's representative seed, from train_explain.csv files,
# and print a LaTeX table row for each.

# Assumes every train_explain.csv has columns:
#     reward_resource, reward_network, reward_security

# Adjust PATHS below to match your actual folder layout.
# """

# import pandas as pd
# from pathlib import Path

# # ---- Configure paths for each algorithm's representative seed ----
# # Based on your established representative-seed choices:
# #   DQN seed0, PPO seed0, A2C seed4, Envelope seed3, EUPG seed2
# PATHS = {
#     "DQN":      Path(r"trained_models/dqn_model_seed0/train_explain.csv"),
#     "PPO":      Path(r"trained_models/ppo_model_seed0/train_explain.csv"),
#     "A2C":      Path(r"trained_models/a2c_model_seed4/train_explain.csv"),
#     "Envelope": Path(r"trained_models/envelope_model_seed3/train_explain.csv"),
#     "EUPG":     Path(r"trained_models/eupg_model_seed2/train_explain.csv"),
# }

# REWARD_COLS = {
#     "Security": "reward_security",
#     "Network":  "reward_network",
#     "Resource": "reward_resource",
# }


# def compute_means(csv_path: Path) -> dict:
#     """Read one train_explain.csv and return mean reward components."""
#     df = pd.read_csv(csv_path, usecols=list(REWARD_COLS.values()))
#     return {
#         label: df[col].mean()
#         for label, col in REWARD_COLS.items()
#     }


# def main():
#     results = {}
#     for algo, path in PATHS.items():
#         if not path.exists():
#             print(f"[WARN] File not found for {algo}: {path}")
#             continue
#         results[algo] = compute_means(path)

#     # ---- Print plain summary ----
#     print("\n=== Mean per-step reward components ===")
#     for algo, vals in results.items():
#         print(f"{algo:9s}  Security={vals['Security']:.4f}  "
#               f"Network={vals['Network']:.4f}  Resource={vals['Resource']:.4f}")

#     # ---- Print LaTeX table rows ----
#     print("\n=== LaTeX table rows ===")
#     order = ["DQN", "PPO", "A2C", "Envelope", "EUPG"]
#     for algo in order:
#         if algo not in results:
#             print(f"% {algo}: MISSING DATA")
#             continue
#         vals = results[algo]
#         print(f"{algo} & {vals['Security']:.3f} & {vals['Network']:.3f} & {vals['Resource']:.3f} \\\\")


# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np

# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "all_seeds_metrics.csv"

# Optional: save the summary to another CSV file
OUTPUT_CSV = "multiseed_summary.csv"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Map model names to paper names
algorithm_names = {
    "dqn_model": "DQN",
    "ppo_model": "PPO",
    "a2c_model": "A2C",
    "envelope_model": "Envelope",
    "eupg_model": "EUPG",
}

df["Algorithm"] = df["model_name"].map(algorithm_names)

# Check for unknown model names
if df["Algorithm"].isna().any():
    unknown = df.loc[df["Algorithm"].isna(), "model_name"].unique()
    raise ValueError(f"Unknown model_name(s): {unknown}")


# ============================================================
# Compute mean and standard deviation
# ============================================================

summary = (
    df.groupby("Algorithm")
      .agg(
          match_mean=("match_rate", "mean"),
          match_std=("match_rate", lambda x: x.std(ddof=1)),
          entropy_mean=("action_entropy", "mean"),
          entropy_std=("action_entropy", lambda x: x.std(ddof=1)),
          n_seeds=("seed", "count"),
      )
      .reset_index()
)

# Keep the desired algorithm order
order = ["DQN", "PPO", "A2C", "Envelope", "EUPG"]
summary["Algorithm"] = pd.Categorical(
    summary["Algorithm"],
    categories=order,
    ordered=True
)
summary = summary.sort_values("Algorithm")


# ============================================================
# Print results
# ============================================================

print("\nMulti-seed results (mean ± std):")
print("=" * 75)

for _, row in summary.iterrows():

    # Convert match rate from [0, 1] to percentage
    match_mean_pct = row["match_mean"] * 100
    match_std_pct = row["match_std"] * 100

    print(
        f"{row['Algorithm']:10s} | "
        f"Match Rate: {match_mean_pct:.2f} ± {match_std_pct:.2f}% | "
        f"Entropy: {row['entropy_mean']:.3f} ± {row['entropy_std']:.3f}"
    )


# ============================================================
# Create LaTeX-ready table rows
# ============================================================

print("\nLaTeX table rows:")
print("=" * 75)

for _, row in summary.iterrows():

    match_mean_pct = row["match_mean"] * 100
    match_std_pct = row["match_std"] * 100

    print(
        f"{row['Algorithm']} "
        f"& {match_mean_pct:.2f} $\\pm$ {match_std_pct:.2f} "
        f"& {row['entropy_mean']:.3f} $\\pm$ {row['entropy_std']:.3f} \\\\"
    )


# ============================================================
# Save summary CSV
# ============================================================

summary.to_csv(OUTPUT_CSV, index=False)

print(f"\nSummary saved to: {OUTPUT_CSV}")
