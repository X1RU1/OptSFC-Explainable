"""
Nearest-Neighbour Surrogate Validation for SHAP
================================================
Purpose
-------
Validates the approximation made inside `_compute_shap_chosen`
(shap_argmax_explain.py): the `policy_fn` used by KernelExplainer does not
call the original Q-network / policy network. Instead, for any queried
feature vector it looks up the nearest background state (by standardized
Euclidean distance) and returns THAT state's recorded Q/prob vector.

SHAP's additivity property only guarantees that attributions sum to the
output of this lookup function. It does not by itself guarantee that this
output agrees with the state's own true recorded value. This script
measures that gap directly, using only the same CSV data already used for
SHAP -- no access to the original network is required.

Method (mirrors policy_fn exactly)
-----------------------------------
For every state i in the full dataset:
    1. Standardize features using the SAME mean/std as _compute_shap_chosen
       (computed from the full X, not just background).
    2. Find its nearest neighbour in the background set (bg), using the
       same standardized squared-Euclidean distance as policy_fn.
    3. Compare action_matrix[i]  (state i's own recorded Q/prob vector)
       against Y_ref[nn_idx]     (nearest background neighbour's recorded
       Q/prob vector, i.e. what policy_fn(i) would actually return).

Metrics reported
-----------------
    argmax_agreement : fraction of states where the nearest-neighbour
                        vector and the state's own vector select the
                        same action (argmax match)
    mean_abs_error    : mean elementwise |own - neighbour| across all
                        n_actions dimensions
    rmse              : root-mean-squared elementwise error
    self_match_rate   : fraction of states whose nearest neighbour is
                        themselves (distance == 0, i.e. the state IS
                        part of the background sample) -- for these,
                        agreement is trivially exact and is reported
                        separately from the non-background subset.

Usage
-----
    from shap_argmax_explain import _get_features, _background, load_data
    from shap_surrogate_validation import validate_surrogate

    df = load_data("path/to.csv")
    X  = _get_features(df)
    action_matrix = df[ACTION_COLS].values.astype(float)  # whichever target
    result = validate_surrogate(X, action_matrix, tag="dqn_scalar_Q")
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from optsfc.envs.shap_argmax_explain import _background


def validate_surrogate(
    X: np.ndarray,
    action_matrix: np.ndarray,
    tag: str,
    output_dir: str = ".",
    seed_background=None,
) -> dict:
    """
    Measure agreement between each state's own recorded action_matrix row
    and its nearest background neighbour's row, under the exact same
    standardization and background sample used by policy_fn in
    _compute_shap_chosen.

    Parameters
    ----------
    X              : (N, n_features) -- same feature matrix passed to
                     _compute_shap_chosen for this algo/scope
    action_matrix  : (N, n_actions)  -- same target matrix passed to
                     _compute_shap_chosen for this algo/scope
                     (e.g. q_a{i}_scalar, prob_action_{i}, scalar_q_a{i},
                     or q_a{i}_{obj})
    tag            : identifier for output file naming, e.g.
                     "dqn_scalar_Q" or "envelope_Q_security"
    output_dir     : where to write the per-state and summary CSVs
    seed_background: optional (bg, bg_idx) tuple to reuse an existing
                     background sample instead of drawing a new one.
                     Pass the SAME background used in _compute_shap_chosen
                     for this call if you want the validation to reflect
                     the exact SHAP run that was performed, rather than a
                     freshly redrawn sample.

    Returns
    -------
    dict with keys: argmax_agreement, mean_abs_error, rmse,
                     self_match_rate, n_states, n_background
    """
    N, n_actions = action_matrix.shape

    if seed_background is not None:
        bg, bg_idx = seed_background
    else:
        bg, bg_idx = _background(X)

    # Standardization: mirrors _compute_shap_chosen exactly
    # (mean/std computed from the full X, not from bg alone)
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    X_scaled = (X - feat_mean) / feat_std
    bg_scaled = (bg - feat_mean) / feat_std
    Y_ref = action_matrix[bg_idx]  # (n_bg, n_actions), aligned to bg via bg_idx

    dists = cdist(X_scaled, bg_scaled, metric="sqeuclidean")  # (N, n_bg)
    nn_idx = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(N), nn_idx]

    own_vec = action_matrix          # (N, n_actions) -- state's own true value
    nn_vec = Y_ref[nn_idx]           # (N, n_actions) -- what policy_fn would return

    own_argmax = own_vec.argmax(axis=1)
    nn_argmax = nn_vec.argmax(axis=1)
    argmax_match = (own_argmax == nn_argmax)

    abs_err = np.abs(own_vec - nn_vec)         # (N, n_actions)
    mean_abs_error = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())

    # States whose nearest neighbour is themselves (distance ~0): these are
    # background states, where policy_fn returns the state's own true value
    # by construction. Report agreement separately for the non-background
    # subset, since that is where the approximation actually matters.
    is_self_match = nn_dist < 1e-12
    self_match_rate = is_self_match.mean()

    non_bg_mask = ~is_self_match
    if non_bg_mask.sum() > 0:
        argmax_agreement_non_bg = argmax_match[non_bg_mask].mean()
        mae_non_bg = abs_err[non_bg_mask].mean()
        rmse_non_bg = np.sqrt((abs_err[non_bg_mask] ** 2).mean())
    else:
        argmax_agreement_non_bg = np.nan
        mae_non_bg = np.nan
        rmse_non_bg = np.nan

    result = {
        "tag": tag,
        "n_states": N,
        "n_background": len(bg_idx),
        "self_match_rate": self_match_rate,
        "argmax_agreement_overall": argmax_match.mean(),
        "argmax_agreement_non_background": argmax_agreement_non_bg,
        "mean_abs_error_overall": mean_abs_error,
        "mean_abs_error_non_background": mae_non_bg,
        "rmse_overall": rmse,
        "rmse_non_background": rmse_non_bg,
    }

    # Per-state detail, for inspection / plotting if needed
    detail = pd.DataFrame({
        "state_idx": np.arange(N),
        "is_background_state": is_self_match,
        "own_argmax": own_argmax,
        "neighbour_argmax": nn_argmax,
        "argmax_match": argmax_match,
        "mean_abs_error": abs_err.mean(axis=1),
    })

    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, f"surrogate_validation_{tag}_detail.csv")
    detail.to_csv(detail_path, index=False)

    summary_path = os.path.join(output_dir, f"surrogate_validation_{tag}_summary.csv")
    pd.DataFrame([result]).to_csv(summary_path, index=False)

    print(f"[Surrogate validation / {tag}]")
    print(f"  n_states={N}, n_background={len(bg_idx)}, "
          f"self_match_rate={self_match_rate:.3f}")
    print(f"  argmax agreement (non-background states): "
          f"{argmax_agreement_non_bg:.3f}")
    print(f"  mean abs error   (non-background states): {mae_non_bg:.4f}")
    print(f"  rmse             (non-background states): {rmse_non_bg:.4f}")
    print(f"  Saved detail  -> {detail_path}")
    print(f"  Saved summary -> {summary_path}")

    return result


def validate_all(df: pd.DataFrame, algo: str, output_dir: str = "surrogate_validation_outputs"):
    """
    Convenience wrapper: runs validate_surrogate for every SHAP target
    associated with a given algo, using the same feature/target columns
    as shap_argmax_explain.py's per-algorithm runners.

    Call this right after (or instead of) the corresponding run_*_shap
    function to validate that specific run.
    """
    from optsfc.envs.shap_argmax_explain import (
        _get_features,
        ACTION_COLS_SCALAR,
        ACTION_COLS_PROB,
        ENVELOPE_SCALAR_COLS,
        ENVELOPE_OBJECTIVES,
        N_ACTIONS,
    )

    X = _get_features(df)
    os.makedirs(output_dir, exist_ok=True)

    if algo == "DQN":
        action_matrix = df[ACTION_COLS_SCALAR].values.astype(np.float64)
        return {"dqn_scalar_Q": validate_surrogate(X, action_matrix, "dqn_scalar_Q", output_dir)}

    if algo in ("PPO", "A2C", "EUPG"):
        action_matrix = df[ACTION_COLS_PROB].values.astype(np.float64)
        tag = f"{algo.lower()}_policy_prob"
        return {tag: validate_surrogate(X, action_matrix, tag, output_dir)}

    if algo == "Envelope":
        results = {}
        scalar_matrix = df[ENVELOPE_SCALAR_COLS].values.astype(np.float64)
        results["envelope_scalar_Q"] = validate_surrogate(
            X, scalar_matrix, "envelope_scalar_Q", output_dir
        )
        for obj in ENVELOPE_OBJECTIVES:
            obj_cols = [f"q_a{i}_{obj}" for i in range(N_ACTIONS)]
            obj_matrix = df[obj_cols].values.astype(np.float64)
            results[f"envelope_Q_{obj}"] = validate_surrogate(
                X, obj_matrix, f"envelope_Q_{obj}", output_dir
            )
        return results

    raise ValueError(f"Unknown algo: {algo}")


if __name__ == "__main__":
    import argparse
    from optsfc.envs.shap_argmax_explain import load_data

    parser = argparse.ArgumentParser(
        description="Validate the nearest-neighbour surrogate used inside SHAP's policy_fn."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="surrogate_validation_outputs", help="Output directory")
    parser.add_argument(
        "--algo", default=None,
        help="Run only one algorithm (DQN | Envelope | EUPG | PPO | A2C). "
             "Omit to run all algorithms found in the CSV.",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    algos = [args.algo] if args.algo else data["algo"].unique().tolist()

    for algo in algos:
        algo_df = data[data["algo"] == algo].reset_index(drop=True)
        if algo_df.empty:
            print(f"[WARNING] No rows for algo='{algo}', skipping.")
            continue
        print(f"\n{'=' * 60}\nSurrogate validation: {algo}\n{'=' * 60}")
        validate_all(algo_df, algo, args.output)