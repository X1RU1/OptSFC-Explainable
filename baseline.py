"""
Baseline evaluation: No-MTD and Random policies.

Mirrors eval_dqn / eval_ppo / eval_envelope in optsfc/envs/mo_fiveg_mdp.py
as closely as possible, so the resulting reward components are directly
comparable to the RL agents' eval_explain.csv results.

Key design choices, matching your existing pipeline:
  - non_MORL=False -> env.step() returns the 3-component reward vector
    [resource, network, security] directly (same quantity that ends up
    in reward_resource / reward_network / reward_security in your
    RL agents' explain CSVs, via reward_noScalar).
  - eval_seed = seed + 10000, set_global_seed(eval_seed) -> same seeding
    convention as eval_dqn/eval_ppo/eval_envelope/eval_eupg.
  - "No-MTD" policy = action 0 every step (this is literally the
    "nothing" branch already implemented in eval_mo_reward_conditioned
    in optsfc/envs/mo_fiveg.py).
  - "Random" policy = env.action_space.sample() filtered through
    is_action_possible(), exactly like the "random" branch in the same
    function.
  - No model_for_explain is attached (no RDX/SHAP machinery), since
    baselines have no Q-network -- we log reward_noScalar ourselves.

Usage:
    python run_baselines.py
Produces:
    ./trained_models/no_mtd_baseline/explain.csv
    ./trained_models/random_baseline/explain.csv
and prints mean reward components + LaTeX rows for Table 1.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from pathlib import Path

from optsfc.envs.mo_fiveg_mdp import MOfiveG_net, set_global_seed
from optsfc.envs.short_simulated_testbed import is_action_possible

# ---------------------------------------------------------------
# CONFIG -- match what you used for RL eval in your run script
# ---------------------------------------------------------------
SEED = 3                    # pick any seed id; only used to derive eval_seed
EVAL_STEPS = 100000          # match eval_steps used for DQN/PPO/A2C/Envelope/EUPG
BUDGET_RESET = "daily"       # match budget_reset used for RL eval
OUTPUT_DIR = Path("./trained_models")


def make_baseline_env(seed):
    """Mirrors _make_eval_env(), but non_MORL=False so step() returns the
    3-component reward vector directly (no scalarization needed)."""
    eval_seed = seed + 10000
    set_global_seed(eval_seed)
    env = MOfiveG_net("MlpPolicy", budget_reset=BUDGET_RESET, non_MORL=False)
    env.action_space.seed(eval_seed)
    env.reset(seed=eval_seed)
    return env


def sample_valid_random_action(env):
    """Same logic as the 'random' branch in eval_mo_reward_conditioned:
    resample until the action is actually executable."""
    while True:
        action = env.action_space.sample()
        if is_action_possible(env.environment, action)[0]:
            return action


def rollout(env, policy_fn, eval_steps):
    """Runs eval_steps environment steps, logging the 3-component reward
    vector per step. Mirrors the step loop in eval_dqn/eval_envelope."""
    rows = []
    obs, _ = env.reset()
    for step in range(eval_steps):
        action = policy_fn(env)
        obs, reward, done, truncated, info = env.step(action)
        # non_MORL=False -> reward IS the np.array([resource, network, security])
        rows.append({
            "step": step,
            "action": action,
            "reward_resource": float(reward[0]),
            "reward_network":  float(reward[1]),
            "reward_security": float(reward[2]),
        })
        if done or truncated:
            obs, _ = env.reset()
    return pd.DataFrame(rows)


def main():
    policies = {
        "no_mtd": lambda env: 0,
        "random": sample_valid_random_action,
    }

    summary = {}
    for name, policy_fn in policies.items():
        env = make_baseline_env(SEED)
        df = rollout(env, policy_fn, EVAL_STEPS)

        save_root = OUTPUT_DIR / f"{name}_baseline"
        save_root.mkdir(parents=True, exist_ok=True)
        out_path = save_root / "explain.csv"
        df.to_csv(out_path, index=False)
        print(f"[{name}] saved {len(df)} rows -> {out_path}")

        summary[name] = {
            "Security": df["reward_security"].mean(),
            "Network":  df["reward_network"].mean(),
            "Resource": df["reward_resource"].mean(),
        }

    print("\n=== Baseline mean per-step reward components ===")
    for name, vals in summary.items():
        print(f"{name:8s}  Security={vals['Security']:.4f}  "
              f"Network={vals['Network']:.4f}  Resource={vals['Resource']:.4f}")

    print("\n=== LaTeX rows (paste into Table 1) ===")
    label_map = {"no_mtd": "No-MTD", "random": "Random"}
    for name, vals in summary.items():
        print(f"{label_map[name]} & {vals['Security']:.3f} & "
              f"{vals['Network']:.3f} & {vals['Resource']:.3f} \\\\")


if __name__ == "__main__":
    main()