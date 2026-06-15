"""
SHAP Analysis for RL Decision Explanation
==========================================
Purpose
-------
Compute SHAP feature attributions to answer:
    "Which objective-related factors most influence the policy?"

This is a preparatory step for SILVER (Shapley value-based Interpretable
policy Via Explanation Regression), which builds a global interpretable
surrogate policy from per-state Shapley vectors Φ_s.

Methodology (aligned with SILVER paper)
-----------------------------------------
The paper uses shap.DeepExplainer directly on the policy network:

    shap_values shape: (N, n_features, n_actions)
    shap_chosen shape: (N, n_features)   ← shap_values[i, :, chosen_action[i]]

Since we operate on CSV (no network objects), we replicate the same logic
using shap.KernelExplainer with a lookup function that maps X → all-action
Q/prob vector, then slice the chosen-action SHAP afterward.

    Step 1: f(X) → (N, n_actions)  [Q values or probabilities, all actions]
    Step 2: KernelExplainer produces shap_values: (N, n_features, n_actions)
    Step 3: shap_chosen = shap_values[range(N), :, chosen_actions]
                        → shape (N, n_features) = Φ_s per state

Feature scope
-------------
Only the 5 objective-related state features are included.
Constraint-related features are excluded (monotonic depletion signals
that dominate SHAP magnitude without reflecting true policy drivers).

    Category   Feature                 Column
    --------   -------                 ------
    Resource   Mean MTD overhead       feat_mean_mtd_overhead
    Network    Mean network penalty    feat_mean_network_penalty
    Network    Max  network penalty    feat_max_network_penalty
    Security   Mean security penalty   feat_mean_security_penalty
    Security   Max  security penalty   feat_max_security_penalty

SHAP target per algorithm
--------------------------
    DQN       → all-action scalar Q vector,         shape (N, 12)
    PPO/A2C/
    EUPG      → all-action softmax prob vector,      shape (N, 12)
    Envelope  → TWO independent SHAP passes:

                (1) Scalarized Q vector (scalar_q_a0 … scalar_q_a11),
                    shape (N, 12), chosen_actions = argmax(scalar_q_a{i}).
                    This mirrors run_dqn_shap exactly (same target type:
                    an all-action scalar value vector, greedy argmax).
                    → shap_envelope_scalar_Q.csv
                    This is the Φ_s consumed by SILVER for Envelope.

                (2) Per-objective Q vectors (resource/network/security),
                    one independent SHAP run each, all sliced at the SAME
                    chosen_actions = argmax(scalar_q_a{i}) as in (1).
                    → shap_envelope_Q_{resource,network,security}.csv
                    Diagnostic only — feeds shap_envelope_objective_influence.csv,
                    NOT used by SILVER.

Output files
------------
    shap_<tag>.csv         — Φ_s: per-state chosen-action SHAP, shape (N, n_features)
    shap_<tag>_summary.csv — mean |SHAP| and mean signed SHAP per feature

    Envelope only:
    shap_envelope_scalar_Q.csv (+ _summary.csv)
                           — Φ_s for the scalarized Q the policy actually
                             maximizes. THIS is the file SILVER consumes
                             (mirrors shap_dqn_scalar_Q.csv for DQN).
    shap_envelope_Q_{resource,network,security}.csv (+ _summary.csv each)
                           — per-objective Φ_s, diagnostic only
                             (NOT consumed by SILVER)
    shap_envelope_objective_influence.csv
                           — weighted influence share per objective (%),
                             derived from the per-objective Φ_s above
"""

import os
import warnings

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Objective-related features only (5 total)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "feat_mean_mtd_overhead",       # Resource: MTD operation cost
    "feat_mean_network_penalty",    # Network:  mean penalty
    "feat_max_network_penalty",     # Network:  max  penalty
    "feat_mean_security_penalty",   # Security: mean penalty
    "feat_max_security_penalty",    # Security: max  penalty
]

N_ACTIONS = 12
ACTION_COLS_PROB     = [f"prob_action_{i}" for i in range(N_ACTIONS)]
ACTION_COLS_SCALAR   = [f"q_a{i}_scalar"    for i in range(N_ACTIONS)]
ENVELOPE_SCALAR_COLS = [f"scalar_q_a{i}"    for i in range(N_ACTIONS)]
ENVELOPE_OBJECTIVES  = ["resource", "network", "security"]

# Envelope reward weights [resource, network, security]
REWARDS_COEFF = [0.4, 0.3, 0.3]

# KernelExplainer background sample size (1% of data, mirrors paper's DeepExplainer)
BACKGROUND_RATIO = 0.01
BACKGROUND_MIN   = 50

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _get_features(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLS].values.astype(np.float64)


def _background(X: np.ndarray) -> np.ndarray:
    """Select background dataset (1% of data, min BACKGROUND_MIN rows)."""
    n = max(BACKGROUND_MIN, int(len(X) * BACKGROUND_RATIO))
    n = min(n, len(X))
    idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)
    return X[idx]


# ---------------------------------------------------------------------------
# Core SHAP computation (paper-aligned)
# ---------------------------------------------------------------------------

def _compute_shap_chosen(
    X: np.ndarray,
    action_matrix: np.ndarray,
    chosen_actions: np.ndarray,
) -> np.ndarray:
    """
    Replicate the paper's DeepExplainer logic using KernelExplainer.

    Parameters
    ----------
    X              : (N, n_features) — objective-related state features
    action_matrix  : (N, n_actions)  — Q values or probabilities for all actions
    chosen_actions : (N,)            — index of the action actually taken

    Returns
    -------
    shap_chosen : (N, n_features) — Φ_s, the per-state Shapley vector
                  corresponding to the chosen action (same as paper's
                  shap_values[sample_idx, :, actions])
    """
    bg = _background(X)

    # f maps a feature matrix (M, n_features) → (M, n_actions)
    # KernelExplainer calls f on subsets of X; we look up the precomputed
    # action values by nearest neighbour in feature space (exact match for
    # rows that exist in X, interpolated otherwise via the lookup table).
    #
    # Implementation: build a lookup from X rows → action_matrix rows.
    # For unseen interpolated points (kernel perturbations), we use the
    # closest row in X by L2 distance.
    X_ref = X
    Y_ref = action_matrix

    def policy_fn(x_subset: np.ndarray) -> np.ndarray:
        # x_subset: (M, n_features) — perturbed feature rows from KernelExplainer
        # Find nearest neighbour in X_ref for each perturbed row
        dists = np.sum((x_subset[:, None, :] - X_ref[None, :, :]) ** 2, axis=2)
        nn_idx = np.argmin(dists, axis=1)
        return Y_ref[nn_idx]  # (M, n_actions)

    explainer = shap.KernelExplainer(policy_fn, bg)
    sv_list   = explainer.shap_values(X, nsamples="auto", silent=True)

    # KernelExplainer multi-output returns either:
    #   (a) list of n_actions arrays each (N, n_features)  → np.array gives (n_actions, N, n_features)
    #   (b) single array (N, n_features, n_actions)        → already correct
    sv_arr = np.array(sv_list)  # stack everything into one array first

    N, n_feat = X.shape
    n_act     = action_matrix.shape[1]

    if sv_arr.shape == (N, n_feat, n_act):
        sv_3d = sv_arr                          # already (N, n_features, n_actions)
    elif sv_arr.shape == (n_act, N, n_feat):
        sv_3d = sv_arr.transpose(1, 2, 0)       # → (N, n_features, n_actions)
    elif sv_arr.shape == (N, n_act, n_feat):
        sv_3d = sv_arr.transpose(0, 2, 1)       # → (N, n_features, n_actions)
    else:
        raise ValueError(
            f"Unexpected SHAP array shape {sv_arr.shape}. "
            f"Expected one of: ({N},{n_feat},{n_act}), ({n_act},{N},{n_feat}), ({N},{n_act},{n_feat})"
        )

    # Slice chosen action — mirrors paper: shap_values[sample_idx, :, actions]
    sample_idx  = np.arange(N)
    shap_chosen = sv_3d[sample_idx, :, chosen_actions]  # (N, n_features)
    return shap_chosen


def _save_result(shap_chosen: np.ndarray, tag: str, output_dir: str) -> None:
    """Save Φ_s matrix and per-feature summary."""
    # Raw Φ_s
    df_out = pd.DataFrame(shap_chosen, columns=FEATURE_COLS)
    raw_path = os.path.join(output_dir, f"shap_{tag}.csv")
    df_out.to_csv(raw_path, index=False)
    print(f"  Saved Φ_s → {raw_path}")

    # Summary
    summary = pd.DataFrame({
        "feature":       FEATURE_COLS,
        "mean_abs_shap": np.abs(shap_chosen).mean(axis=0),
        "mean_shap":     shap_chosen.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    summary_path = os.path.join(output_dir, f"shap_{tag}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Summary   → {summary_path}")


# ---------------------------------------------------------------------------
# Per-algorithm runners
# ---------------------------------------------------------------------------

def run_dqn_shap(df: pd.DataFrame, output_dir: str = "."):
    """
    DQN — all-action scalar Q vector → chosen-action SHAP slice.

    action_matrix shape: (N, 12)  — q_a0_scalar … q_a11_scalar (signed)
    chosen_actions: argmax Q      — greedy action, explaining policy not execution trace
    """
    print("[DQN SHAP] Computing (all-action Q → chosen-action SHAP)...")
    missing = [c for c in ACTION_COLS_SCALAR if c not in df.columns]
    if missing:
        raise ValueError(f"DQN scalar Q columns missing: {missing}")

    X              = _get_features(df)
    action_matrix  = df[ACTION_COLS_SCALAR].values.astype(np.float64)  # (N, 12)
    chosen_actions = action_matrix.argmax(axis=1)                       # greedy policy: argmax Q

    shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
    _save_result(shap_chosen, "dqn_scalar_Q", output_dir)
    return shap_chosen, X


def run_envelope_scalar_shap(df: pd.DataFrame, output_dir: str = "."):
    """
    Envelope — all-action scalarized Q vector → chosen-action SHAP slice.

    Structurally identical to run_dqn_shap: scalar_q_a{i} is an all-action
    scalar value vector (the weighted sum w·Q_vec already stored in the CSV),
    so it is treated exactly like DQN's q_ai_scalar.

    action_matrix shape: (N, 12)  — scalar_q_a0 … scalar_q_a11 (signed)
    chosen_actions: argmax(scalar_q_a{i}) — greedy action, the action the
                    Envelope policy actually selects.

    This is the Φ_s consumed by SILVER for Envelope (shap_envelope_scalar_Q.csv).
    """
    print("[Envelope SHAP] Computing (scalarized Q → chosen-action SHAP, for SILVER)...")
    missing = [c for c in ENVELOPE_SCALAR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Envelope scalar Q columns missing: {missing}")

    X              = _get_features(df)
    action_matrix  = df[ENVELOPE_SCALAR_COLS].values.astype(np.float64)  # (N, 12)
    chosen_actions = action_matrix.argmax(axis=1)                         # greedy policy: argmax scalar Q

    shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
    _save_result(shap_chosen, "envelope_scalar_Q", output_dir)
    return shap_chosen, X


def run_envelope_shap(df: pd.DataFrame, output_dir: str = ".") -> dict:
    """
    Envelope — one SHAP run per objective (resource / network / security).

    For each objective k:
        action_matrix: (N, 12) — q_a0_<obj> … q_a11_<obj> (signed)

    chosen_actions is shared across all three objectives and equals
    argmax(scalar_q_a{i}) — the same action selected in
    run_envelope_scalar_shap — so all per-objective Φ_s here explain the
    SAME chosen action, just w.r.t. different Q functions.

    These three Φ_s matrices are diagnostic — they answer "what drives each
    objective's Q-value at the chosen action" and feed
    shap_envelope_objective_influence.csv. They are NOT the Φ_s consumed by
    SILVER (see run_envelope_scalar_shap for that).
    """
    print("[Envelope SHAP] Computing (per-objective Q → chosen-action SHAP)...")

    # Chosen action = argmax of scalarized Q = w·Q_vec
    # scalar_q_a{i} is already the weighted sum stored in CSV
    missing_scalar = [c for c in ENVELOPE_SCALAR_COLS if c not in df.columns]
    if missing_scalar:
        raise ValueError(f"Envelope scalar Q columns missing: {missing_scalar}")
    chosen_actions = df[ENVELOPE_SCALAR_COLS].values.astype(np.float64).argmax(axis=1)  # greedy policy

    results = {}
    for obj in ENVELOPE_OBJECTIVES:
        obj_cols = [f"q_a{i}_{obj}" for i in range(N_ACTIONS)]
        missing  = [c for c in obj_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Envelope columns missing for '{obj}': {missing}")

        X             = _get_features(df)
        action_matrix = df[obj_cols].values.astype(np.float64)  # (N, 12)

        shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
        _save_result(shap_chosen, f"envelope_Q_{obj}", output_dir)
        results[obj] = shap_chosen
        print(f"[Envelope SHAP] Objective '{obj}' done.")

    return results


def run_envelope_objective_influence(
    envelope_shap_results: dict,
    rewards_coeff: list,
    output_dir: str = ".",
) -> pd.DataFrame:
    """
    Aggregate per-objective Φ_s into a weighted influence-share table.

    For each objective k with reward weight w_k:
        weighted_influence_k = w_k * mean( |Φ_s^k|.sum(axis=1) )

    Answers: "Does resource, network, or security drive Envelope most?"

    This is a diagnostic summary for the thesis discussion and is
    independent of the scalarized Φ_s consumed by SILVER
    (shap_envelope_scalar_Q.csv).
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

    out_path = os.path.join(output_dir, "shap_envelope_objective_influence.csv")
    result.to_csv(out_path, index=False)
    print(f"[Envelope] Objective influence → {out_path}")
    print(result.to_string(index=False))
    return result


def run_policy_prob_shap(df: pd.DataFrame, algo: str, output_dir: str = "."):
    """
    PPO / A2C / EUPG — all-action softmax prob vector → chosen-action SHAP.

    action_matrix shape: (N, 12)  — prob_action_0 … prob_action_11
    chosen_actions: argmax prob   — greedy action, explaining policy not execution trace
    """
    assert algo in ("PPO", "A2C", "EUPG"), f"Unexpected algo: {algo}"
    print(f"[{algo} SHAP] Computing (all-action prob → chosen-action SHAP)...")
    missing = [c for c in ACTION_COLS_PROB if c not in df.columns]
    if missing:
        raise ValueError(f"{algo} prob columns missing: {missing}")

    X              = _get_features(df)
    action_matrix  = df[ACTION_COLS_PROB].values.astype(np.float64)  # (N, 12)
    chosen_actions = action_matrix.argmax(axis=1)                     # greedy policy: argmax prob

    shap_chosen = _compute_shap_chosen(X, action_matrix, chosen_actions)
    _save_result(shap_chosen, f"{algo.lower()}_policy_prob", output_dir)
    return shap_chosen, X


# ---------------------------------------------------------------------------
# Envelope full pipeline
# ---------------------------------------------------------------------------

def _run_envelope_full(df: pd.DataFrame, output_dir: str = "."):
    # Pass 1: scalarized Q → Φ_s for SILVER (mirrors run_dqn_shap)
    run_envelope_scalar_shap(df, output_dir)

    # Pass 2 (x3): per-objective Q → diagnostic Φ_s, feeds influence table
    results = run_envelope_shap(df, output_dir)
    run_envelope_objective_influence(results, REWARDS_COEFF, output_dir)

    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALGO_DISPATCHER = {
    "DQN":      lambda df, out: run_dqn_shap(df, out),
    "Envelope": lambda df, out: _run_envelope_full(df, out),
    "EUPG":     lambda df, out: run_policy_prob_shap(df, "EUPG", out),
    "PPO":      lambda df, out: run_policy_prob_shap(df, "PPO", out),
    "A2C":      lambda df, out: run_policy_prob_shap(df, "A2C", out),
}


def run_all(df: pd.DataFrame, output_dir: str = "shap_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    for algo in df["algo"].unique():
        if algo not in ALGO_DISPATCHER:
            print(f"[WARNING] Unknown algo '{algo}', skipping.")
            continue
        print(f"\n{'='*60}\nCustomized SHAP: {algo}\n{'='*60}")
        algo_df = df[df["algo"] == algo].reset_index(drop=True)
        ALGO_DISPATCHER[algo](algo_df, output_dir)
    print(f"\n✓ All SHAP analyses complete. Outputs in: {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHAP analysis for RL algorithms.")
    parser.add_argument("--input",  required=True, help="Path to input CSV")
    parser.add_argument("--output", default="shap_outputs", help="Output directory")
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
        ALGO_DISPATCHER[args.algo](algo_df, args.output)
    else:
        run_all(data, args.output)