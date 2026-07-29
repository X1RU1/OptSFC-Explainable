from optsfc import MOfiveG_net, train, train_Envelope, train_eupg, eval_dqn, eval_ppo, eval_a2c, eval_envelope, eval_eupg
import torch

seeds = [0, 1, 2, 3, 4]

# ── DQN ────────────────────────────────────────────────────────────────────
for seed in seeds:
    train(
        agent_type="DQN",
        policy="MlpPolicy",
        total_timesteps=5000,
        model_name="dqn_model",
        log_dir=f"./logs/dqn_seed{seed}/",
        budget_reset="episodic",
        seed=seed,
        dqn_learning_starts=200,   # confirm/adjust this value and justify it in the report
    )
    eval_dqn(
        model_name="dqn_model",
        seed=seed,
        eval_steps=5000,
        budget_reset="episodic",
    )


# ── PPO ────────────────────────────────────────────────────────────────────
for seed in seeds:
    train(
        agent_type="PPO",
        policy="MlpPolicy",
        total_timesteps=5000,
        model_name="ppo_model",
        log_dir=f"./logs/ppo_seed{seed}/",
        budget_reset="episodic",
        seed=seed,
    )
    eval_ppo(
        model_name="ppo_model",
        seed=seed,
        eval_steps=5000,          # unified with other algorithms; see note below
        budget_reset="episodic",
    )


# ── A2C ────────────────────────────────────────────────────────────────────
for seed in seeds:
    train(
        agent_type="A2C",
        policy="MlpPolicy",
        total_timesteps=5000,
        model_name="a2c_model",
        log_dir=f"./logs/a2c_seed{seed}/",
        budget_reset="episodic",
        seed=seed,
    )
    eval_a2c(
        model_name="a2c_model",
        seed=seed,
        eval_steps=5000,
        budget_reset="episodic",
    )


# ── Envelope ───────────────────────────────────────────────────────────────
for seed in seeds:
    train_Envelope(
        total_timesteps=5000,
        model_name="envelope_model",
        budget_reset="episodic",
        seed=seed,
    )
    eval_envelope(
        model_name="envelope_model",
        seed=seed,
        eval_steps=5000,
        budget_reset="episodic",
    )


# ── EUPG ───────────────────────────────────────────────────────────────────
for seed in seeds:
    train_eupg(
        total_timesteps=5000,
        model_name="eupg_model",
        budget_reset="episodic",
        seed=seed,
    )
    eval_eupg(
        model_name="eupg_model",
        seed=seed,
        eval_steps=5000,
        budget_reset="episodic",
    )