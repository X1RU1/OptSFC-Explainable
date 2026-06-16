"""
SHAP Analysis for RL Decision Explanation (env_action variant)
==============================================================
Purpose
-------
Same as shap_argmax_explain.py, but the "chosen action" used to slice the SHAP
tensor is taken from the column `env_action` — the action the agent
actually executed in the environment — rather than argmax(Q/prob).

Why two variants?
-----------------
    argmax variant  — Explains the *policy's greedy intent*.
                      For stochastic policies (PPO/A2C/EUPG), argmax prob
                      may differ from what was actually sampled.
                      For DQN the two are usually identical (ε≈0 at eval).

    env_action variant — Explains the *executed trajectory*.
                         More faithful to what the agent actually did;
                         useful when you want attributions tied to real
                         observed outcomes (rewards, constraint violations).

Column requirement
------------------
    Every input CSV must contain an integer column `env_action` in [0, 11]
    representing the action index actually taken at each timestep.

Everything else (feature scope, output format, SHAP computation) is
identical to shap_argmax_explain.py.  Output tags are suffixed with `_env` to
avoid collisions:

    shap_dqn_scalar_Q_env.csv               (+ _summary.csv)
    shap_envelope_scalar_Q_env.csv          (+ _summary.csv)
    shap_envelope_Q_{resource,network,security}_env.csv (+ _summary.csv)
    shap_envelope_objective_influence_env.csv
    shap_{ppo,a2c,eupg}_policy_prob_env.csv (+ _summary.csv)
"""

import os
import warnings

import numpy as np
import pandas as pd
import shap

# Re-use constants and helpers from the argmax variant
from shap_argmax_explain import (
    FEATURE_COLS,
    N_ACTIONS,
    ACTION_COLS_PROB,
    ACTION_COLS_SCALAR,
    ENVELOPE_SCALAR_COLS,
    ENVELOPE_OBJECTIVES,
    REWARDS_COEFF,
    _get_features,
    _background,
    _compute_shap_chosen,
    _save_result,
    load_data,
    run_envelope_objective_influence,
)

warnings.filterwarnings("ignore")

ENV_ACTION_COL = "env_action"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _get_env_actions(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the env_action column and validate its contents.

    Raises
    ------
    ValueError  if the column is absent or contains out-of-range indices.
    """
    if ENV_ACTION_COL not in df.columns:
        raise ValueError(
            f"Column '{ENV_ACTION_COL}' not found. "
            "The env_action variant requires the actual executed action index "
            "to be present in the input CSV."
        )
    actions = df[ENV_ACTION_COL].values.astype(int)
    if actions.min() < 0 or actions.max() >= N_ACTIONS:
        raise ValueError(
            f"'{ENV_ACTION_COL}' values must be in [0, {N_ACTIONS - 1}]. "
            f"Got range [{actions.min()}, {actions.max()}]."
        )
    return actions


# ---------------------------------------------------------------------------
# Per-algorithm runners (env_action variant)
# ---------------------------------------------------------------------------

def run_dqn_shap_env(df: pd.DataFrame, output_dir: str = "."):
    """
    DQN — all-action scalar Q vector → env_action SHAP slice.

    chosen_actions: env_action column (actual action executed in env)
    """
    print("[DQN SHAP / env_action] Computing (all-action Q → env_action SHAP)...")
    missing = [c for c in ACTION_COLS_SCALAR if c not in df.columns]
    if missing:
        raise ValueError(f"DQN scalar Q columns missing: {missing}")

    X              = _get_features(df)
    action_matrix  = df[ACTION_COLS_SCALAR].values.astype(np.float64)   # (N, 12)
    chosen_actions = _get_env_actions(df)                                # env_action

    shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
    _save_result(shap_chosen, "dqn_scalar_Q_env", output_dir)
    return shap_chosen, X


def run_envelope_scalar_shap_env(df: pd.DataFrame, output_dir: str = "."):
    """
    Envelope — scalarized Q vector → env_action SHAP slice (for SILVER).

    chosen_actions: env_action column — same action used across all
    three per-objective passes below, so all Φ_s explain the SAME step.
    """
    print("[Envelope SHAP / env_action] Computing (scalarized Q → env_action SHAP, for SILVER)...")
    missing = [c for c in ENVELOPE_SCALAR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Envelope scalar Q columns missing: {missing}")

    X              = _get_features(df)
    action_matrix  = df[ENVELOPE_SCALAR_COLS].values.astype(np.float64)  # (N, 12)
    chosen_actions = _get_env_actions(df)                                 # env_action

    shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
    _save_result(shap_chosen, "envelope_scalar_Q_env", output_dir)
    return shap_chosen, X


def run_envelope_shap_env(df: pd.DataFrame, output_dir: str = ".") -> dict:
    """
    Envelope — one SHAP run per objective (resource / network / security),
    all sliced at env_action.

    chosen_actions is shared across all three objectives (same env_action),
    so the per-objective Φ_s explain the same executed action from three
    different Q-function perspectives.  Diagnostic only; NOT consumed by SILVER.
    """
    print("[Envelope SHAP / env_action] Computing (per-objective Q → env_action SHAP)...")
    chosen_actions = _get_env_actions(df)

    results = {}
    for obj in ENVELOPE_OBJECTIVES:
        obj_cols = [f"q_a{i}_{obj}" for i in range(N_ACTIONS)]
        missing  = [c for c in obj_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Envelope columns missing for '{obj}': {missing}")

        X             = _get_features(df)
        action_matrix = df[obj_cols].values.astype(np.float64)  # (N, 12)

        shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
        _save_result(shap_chosen, f"envelope_Q_{obj}_env", output_dir)
        results[obj] = shap_chosen
        print(f"[Envelope SHAP / env_action] Objective '{obj}' done.")

    return results


def run_policy_prob_shap_env(df: pd.DataFrame, algo: str, output_dir: str = "."):
    """
    PPO / A2C / EUPG — all-action softmax prob vector → env_action SHAP.

    chosen_actions: env_action column — the action that was *sampled* from
    the stochastic policy, which may differ from argmax(prob).  This is
    especially meaningful here because these algorithms explicitly explore.
    """
    assert algo in ("PPO", "A2C", "EUPG"), f"Unexpected algo: {algo}"
    print(f"[{algo} SHAP / env_action] Computing (all-action prob → env_action SHAP)...")
    missing = [c for c in ACTION_COLS_PROB if c not in df.columns]
    if missing:
        raise ValueError(f"{algo} prob columns missing: {missing}")

    X              = _get_features(df)
    action_matrix  = df[ACTION_COLS_PROB].values.astype(np.float64)  # (N, 12)
    chosen_actions = _get_env_actions(df)                             # env_action

    shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
    _save_result(shap_chosen, f"{algo.lower()}_policy_prob_env", output_dir)
    return shap_chosen, X


# ---------------------------------------------------------------------------
# Envelope full pipeline (env_action variant)
# ---------------------------------------------------------------------------

def _run_envelope_full_env(df: pd.DataFrame, output_dir: str = "."):
    # Pass 1: scalarized Q → Φ_s for SILVER
    run_envelope_scalar_shap_env(df, output_dir)

    # Pass 2 (x3): per-objective Q → diagnostic Φ_s, feeds influence table
    results = run_envelope_shap_env(df, output_dir)

    # Reuse the same aggregation logic with env-suffixed output name
    _run_envelope_objective_influence_env(results, REWARDS_COEFF, output_dir)

    return results


def _run_envelope_objective_influence_env(
    envelope_shap_results: dict,
    rewards_coeff: list,
    output_dir: str = ".",
) -> pd.DataFrame:
    """
    Same aggregation as run_envelope_objective_influence, but writes to
    shap_envelope_objective_influence_env.csv.
    """
    assert len(rewards_coeff) == len(ENVELOPE_OBJECTIVES)

    weighted = {
        obj: w * np.abs(envelope_shap_results[obj].sum(axis=1)).mean()
        for obj, w in zip(ENVELOPE_OBJECTIVES, rewards_coeff)
    }
    total  = sum(weighted.values())
    result = pd.DataFrame({
        "objective":          ENVELOPE_OBJECTIVES,
        "weight":             rewards_coeff,
        "mean_weighted_shap": [weighted[o] for o in ENVELOPE_OBJECTIVES],
        "influence_pct":      [weighted[o] / total * 100 for o in ENVELOPE_OBJECTIVES],
    }).sort_values("influence_pct", ascending=False)

    out_path = os.path.join(output_dir, "shap_envelope_objective_influence_env.csv")
    result.to_csv(out_path, index=False)
    print(f"[Envelope / env_action] Objective influence → {out_path}")
    print(result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALGO_DISPATCHER_ENV = {
    "DQN":      lambda df, out: run_dqn_shap_env(df, out),
    "Envelope": lambda df, out: _run_envelope_full_env(df, out),
    "EUPG":     lambda df, out: run_policy_prob_shap_env(df, "EUPG", out),
    "PPO":      lambda df, out: run_policy_prob_shap_env(df, "PPO", out),
    "A2C":      lambda df, out: run_policy_prob_shap_env(df, "A2C", out),
}


def run_all_env(df: pd.DataFrame, output_dir: str = "shap_env_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    for algo in df["algo"].unique():
        if algo not in ALGO_DISPATCHER_ENV:
            print(f"[WARNING] Unknown algo '{algo}', skipping.")
            continue
        print(f"\n{'='*60}\nenv_action SHAP: {algo}\n{'='*60}")
        algo_df = df[df["algo"] == algo].reset_index(drop=True)
        ALGO_DISPATCHER_ENV[algo](algo_df, output_dir)
    print(f"\n✓ All env_action SHAP analyses complete. Outputs in: {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SHAP analysis (env_action variant) for RL algorithms."
    )
    parser.add_argument("--input",  required=True, help="Path to input CSV")
    parser.add_argument("--output", default="shap_env_outputs", help="Output directory")
    parser.add_argument(
        "--algo", default=None,
        help="Run only one algorithm (DQN | Envelope | EUPG | PPO | A2C). "
             "Omit to run all algorithms found in the CSV.",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    print(f"Loaded {len(data)} rows, algorithms: {data['algo'].unique().tolist()}")

    if args.algo:
        os.makedirs(args.output, exist_ok=True)
        algo_df = data[data["algo"] == args.algo].reset_index(drop=True)
        if algo_df.empty:
            raise ValueError(f"No rows found for algo='{args.algo}'")
        ALGO_DISPATCHER_ENV[args.algo](algo_df, args.output)
    else:
        run_all_env(data, args.output) 