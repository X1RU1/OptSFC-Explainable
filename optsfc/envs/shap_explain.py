"""
SHAP Analysis for RL Decision Explanation
==========================================
Two-layer SHAP design:

  SHAP (for all):
    X = 23 state features (feat_*)
    Y = match (0/1) for every action pair
    → Binary classification: did the agent choose the "better" action?

  SHAP (customized) per algorithm:
    Envelope  → per-objective Q (resource / network / security) — signed Q value
    EUPG      → policy probability π(a_i, a_j)                  — probability ∈ (0,1)
    DQN       → scalar Q                                         — signed Q value
    PPO       → policy probability                               — probability ∈ (0,1)
    A2C       → policy probability                               — probability ∈ (0,1)

  Note on sign:
    Q values (DQN, Envelope) are kept signed — positive/negative Q is meaningful.
    Probabilities (EUPG, PPO, A2C) are naturally ∈ (0,1), no transformation needed.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional
import warnings
import os

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    "feat_steps_since_last_mtd",
    "feat_total_ues", "feat_nb_resources",
]  # 23 features

N_ACTIONS = 12
ACTION_COLS_PROB   = [f"prob_action_{i}"   for i in range(N_ACTIONS)]
ACTION_COLS_SCALAR = [f"q_a{i}_scalar"     for i in range(N_ACTIONS)]
ENVELOPE_OBJECTIVES = ["resource", "network", "security"]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLS].copy()


# ---------------------------------------------------------------------------
# Layer 1 — Universal SHAP  (per-action, X=features, Y=match 0/1)
# ---------------------------------------------------------------------------

def run_universal_shap(df: pd.DataFrame, output_dir: str = "."):
    """
    Layer 1: per-action binary classification SHAP.

    For each action_i (0..N_ACTIONS-1), train a separate classifier where:
        X = 23 state features
        Y = 1 if env_action == action_i, else 0

    All 12 classifiers are merged into one CSV with columns:
        action | feat_vim0_cpu | ... | shap_feat_vim0_cpu | ... | match

    One summary CSV ranks features by mean |SHAP| averaged across all actions.
    """
    print("[Universal SHAP] Running per-action classifiers...")
    feats = get_features(df)
    X_raw = feats.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    all_blocks = []

    for action_i in range(N_ACTIONS):
        Y = (df["env_action"].astype(int) == action_i).astype(int).values

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        )
        model.fit(X_scaled, Y)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)
        # GradientBoostingClassifier: shap_values is (n_samples, n_features)
        # older sklearn may return a list [class0, class1] — take class-1
        if isinstance(sv, list):
            sv = sv[1]

        block = pd.DataFrame(X_raw, columns=FEATURE_COLS)
        shap_block = pd.DataFrame(sv, columns=[f"shap_{f}" for f in FEATURE_COLS])
        block = pd.concat([block, shap_block], axis=1)
        block.insert(0, "action", action_i)
        block["match"] = Y
        all_blocks.append(block)

        action_summary = pd.DataFrame({
            "feature":       FEATURE_COLS,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
            "mean_shap":     sv.mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        action_summary_path = os.path.join(output_dir, f"shap_universal_summary_action{action_i}.csv")
        action_summary.to_csv(action_summary_path, index=False)
        print(f"  action {action_i:2d} done  "
              f"(n_positive={Y.sum()}, n_negative={(1 - Y).sum()})")

    # Merged CSV: N_steps × N_actions rows
    merged = pd.concat(all_blocks, ignore_index=True)
    out_path = os.path.join(output_dir, "shap_universal.csv")
    merged.to_csv(out_path, index=False)
    print(f"[Universal SHAP] Saved → {out_path}  "
          f"({len(merged)} rows = {len(df)} steps × {N_ACTIONS} actions)")

    return merged


# ---------------------------------------------------------------------------
# Layer 2 — Customized SHAP per algorithm
# ---------------------------------------------------------------------------

# --- DQN: scalar Q (signed) -------------------------------------------------

def run_dqn_shap(df: pd.DataFrame, output_dir: str = "."):
    """
    Y = Q value of each action, signed (positive/negative Q is meaningful).
    One sample per (step, action), Y = q_a{i}_scalar.
    """
    print("[DQN SHAP] Building dataset...")
    missing = [c for c in ACTION_COLS_SCALAR if c not in df.columns]
    if missing:
        raise ValueError(f"DQN scalar Q columns missing: {missing}")

    feats = get_features(df)
    rows_X, rows_Y = [], []

    for idx, row in df.iterrows():
        x = feats.loc[idx].values
        for i, col in enumerate(ACTION_COLS_SCALAR):
            rows_X.append(x)
            rows_Y.append(row[col])

    X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
    Y = pd.Series(rows_Y, name="scalar_Q")

    sv = _fit_regressor_shap(X, Y)
    _save_shap_result(sv, X, Y, "dqn_scalar_Q", output_dir)
    return sv, X


# --- Envelope: per-objective Q (signed, 3 separate SHAP runs) --------------

def run_envelope_shap(df: pd.DataFrame, output_dir: str = "."):
    """
    Three separate SHAP regressions, one per objective.
    Y = q_a{i}_{objective} signed Q value (positive/negative is meaningful).
    """
    print("[Envelope SHAP] Building dataset (3 objectives)...")
    results = {}
    feats = get_features(df)

    for obj in ENVELOPE_OBJECTIVES:
        obj_cols = [f"q_a{i}_{obj}" for i in range(N_ACTIONS)]
        missing = [c for c in obj_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Envelope columns missing for '{obj}': {missing}")

        rows_X, rows_Y = [], []
        for idx, row in df.iterrows():
            x = feats.loc[idx].values
            for col in obj_cols:
                rows_X.append(x)
                rows_Y.append(row[col])

        X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
        Y = pd.Series(rows_Y, name=f"Q_{obj}")

        sv = _fit_regressor_shap(X, Y)
        _save_shap_result(sv, X, Y, f"envelope_Q_{obj}", output_dir)
        results[obj] = (sv, X)
        print(f"[Envelope SHAP] Objective '{obj}' done.")

    return results


# --- Envelope: objective influence aggregation ------------------------------

def run_envelope_objective_influence(
    envelope_shap_results: dict,
    rewards_coeff: list,
    output_dir: str = ".",
):
    """
    After running per-objective SHAP for Envelope, compute how much each
    objective (resource / network / security) influences the final action
    selection overall.

    Method
    ------
    For each objective obj_k with weight w_k:
        total_shap_per_sample = |sv_k.sum(axis=1)|   # scalar per step×action
        weighted_influence_k  = w_k * total_shap_per_sample

    Then average across all samples and normalise to get percentage shares.

    Args
    ----
    envelope_shap_results : dict returned by run_envelope_shap()
                            keys = "resource" / "network" / "security"
                            values = (sv: np.ndarray, X: pd.DataFrame)
    rewards_coeff         : list of 3 weights [w_resource, w_network, w_security]
    output_dir            : where to write the CSV
    """
    assert len(rewards_coeff) == len(ENVELOPE_OBJECTIVES), (
        "rewards_coeff must have one entry per objective"
    )

    weighted_influences = {}
    for obj, w in zip(ENVELOPE_OBJECTIVES, rewards_coeff):
        sv, _ = envelope_shap_results[obj]
        # Sum all feature SHAP values per sample → total contribution of this
        # objective's Q-signal, then weight by the objective's coefficient
        total_per_sample = np.abs(sv.sum(axis=1))   # shape (n_samples,)
        weighted_influences[obj] = w * total_per_sample

    # Stack into (n_samples, 3) and compute mean per objective
    stacked = np.stack(
        [weighted_influences[obj] for obj in ENVELOPE_OBJECTIVES], axis=1
    )  # (n_samples, 3)

    mean_influence = stacked.mean(axis=0)            # (3,)
    total          = mean_influence.sum()
    pct_influence  = mean_influence / total * 100    # percentage share

    result = pd.DataFrame({
        "objective":          ENVELOPE_OBJECTIVES,
        "weight":             rewards_coeff,
        "mean_weighted_shap": mean_influence,
        "influence_pct":      pct_influence,
    }).sort_values("influence_pct", ascending=False)

    out_path = os.path.join(output_dir, "shap_envelope_objective_influence.csv")
    result.to_csv(out_path, index=False)
    print(f"[Envelope] Objective influence saved → {out_path}")
    print(result.to_string(index=False))
    return result


# --- EUPG: policy probability π(a_i, a_j) -----------------------------------

def run_eupg_shap(df: pd.DataFrame, output_dir: str = "."):
    """
    Y = prob_action_i (softmax probability) for each action.
    One sample per (step, action).
    """
    print("[EUPG SHAP] Building dataset...")
    missing = [c for c in ACTION_COLS_PROB if c not in df.columns]
    if missing:
        raise ValueError(f"EUPG prob columns missing: {missing}")

    feats = get_features(df)
    rows_X, rows_Y = [], []

    for idx, row in df.iterrows():
        x = feats.loc[idx].values
        for col in ACTION_COLS_PROB:
            rows_X.append(x)
            rows_Y.append(row[col])

    X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
    Y = pd.Series(rows_Y, name="policy_prob")

    sv = _fit_regressor_shap(X, Y)
    _save_shap_result(sv, X, Y, "eupg_policy_prob", output_dir)
    return sv, X


# --- PPO / A2C: policy probability ------------------------------------------

def run_policy_prob_shap(df: pd.DataFrame, algo: str, output_dir: str = "."):
    """
    Y = prob_action_i (softmax probability) for each action.
    CSV stores prob_action_i directly — no log or logit transformation needed.
    One sample per (step, action).
    """
    assert algo in ("PPO", "A2C"), f"Expected PPO or A2C, got {algo}"
    print(f"[{algo} SHAP] Building dataset (policy probability)...")
    missing = [c for c in ACTION_COLS_PROB if c not in df.columns]
    if missing:
        raise ValueError(f"{algo} prob columns missing: {missing}")

    feats = get_features(df)
    rows_X, rows_Y = [], []

    for idx, row in df.iterrows():
        x = feats.loc[idx].values
        for col in ACTION_COLS_PROB:
            rows_X.append(x)
            rows_Y.append(row[col])

    X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
    Y = pd.Series(rows_Y, name="policy_prob")

    sv = _fit_regressor_shap(X, Y)
    _save_shap_result(sv, X, Y, f"{algo.lower()}_policy_prob", output_dir)
    return sv, X


# ---------------------------------------------------------------------------
# Shared fitting / saving utilities
# ---------------------------------------------------------------------------

def _fit_regressor_shap(X: pd.DataFrame, Y: pd.Series) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_scaled, Y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values


def _save_shap_result(sv: np.ndarray, X: pd.DataFrame, Y: pd.Series,
                      name: str, output_dir: str):
    result = pd.DataFrame(sv, columns=FEATURE_COLS)
    result[Y.name] = Y.values
    out_path = os.path.join(output_dir, f"shap_{name}.csv")
    result.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    _save_summary(sv, FEATURE_COLS, name,
                  os.path.join(output_dir, f"shap_{name}_summary.csv"))


def _save_summary(sv: np.ndarray, feature_names: list, name: str, out_path: str):
    """
    mean_abs_shap: global importance (magnitude, used for ranking).
    mean_shap:     signed directional bias across all samples.
    """
    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "mean_shap":     sv.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(out_path, index=False)
    print(f"  Summary → {out_path}")


# ---------------------------------------------------------------------------
# Envelope full pipeline (SHAP + objective influence)
# ---------------------------------------------------------------------------

REWARDS_COEFF = [0.4, 0.3, 0.3]   # [resource, network, security]

def _run_envelope_full(df: pd.DataFrame, output_dir: str = "."):
    """Run per-objective SHAP then compute objective influence aggregation."""
    results = run_envelope_shap(df, output_dir)
    run_envelope_objective_influence(results, REWARDS_COEFF, output_dir)
    return results


# ---------------------------------------------------------------------------
# Dispatcher: route each algorithm to the right SHAP function
# ---------------------------------------------------------------------------

ALGO_DISPATCHER = {
    "DQN":      lambda df, out: run_dqn_shap(df, out),
    "Envelope": lambda df, out: _run_envelope_full(df, out),
    "EUPG":     lambda df, out: run_eupg_shap(df, out),
    "PPO":      lambda df, out: run_policy_prob_shap(df, "PPO", out),
    "A2C":      lambda df, out: run_policy_prob_shap(df, "A2C", out),
}


def run_all(df: pd.DataFrame, output_dir: str = "shap_outputs"):
    """
    Run both SHAP layers for all algorithms present in the dataframe.

    Layer 1 (universal) runs once on the full dataset.
    Layer 2 (customized) runs per algorithm on its own subset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Layer 1: universal (all rows together) ---
    run_universal_shap(df, output_dir)

    # --- Layer 2: per algorithm ---
    algos_present = df["algo"].unique()
    for algo in algos_present:
        if algo not in ALGO_DISPATCHER:
            print(f"[WARNING] Unknown algo '{algo}', skipping customized SHAP.")
            continue
        print(f"\n{'='*60}")
        print(f"Customized SHAP: {algo}")
        print('='*60)
        algo_df = df[df["algo"] == algo].reset_index(drop=True)
        ALGO_DISPATCHER[algo](algo_df, output_dir)

    print(f"\n✓ All SHAP analyses complete. Outputs in: {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SHAP analysis for RL algorithms.")
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="shap_outputs", help="Output directory")
    parser.add_argument("--algo",   default=None,
                        help="Run only this algorithm's customized SHAP (optional). "
                             "Choices: DQN, Envelope, EUPG, PPO, A2C")
    args = parser.parse_args()

    data = load_data(args.input)
    print(f"Loaded {len(data)} rows, algorithms: {data['algo'].unique().tolist()}")

    if args.algo:
        # Single algorithm mode
        os.makedirs(args.output, exist_ok=True)
        algo_df = data[data["algo"] == args.algo].reset_index(drop=True)
        if algo_df.empty:
            raise ValueError(f"No rows found for algo='{args.algo}'")
        ALGO_DISPATCHER[args.algo](algo_df, args.output)
    else:
        run_all(data, args.output)