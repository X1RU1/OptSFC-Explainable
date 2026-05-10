from optsfc import MOfiveG_net, train_eupg
import torch

env_eupg = train_eupg(
    total_timesteps=5000,
    model_name="eupg_model",
    budget_reset="episodic"
)
env_eupg.save_explanations("eupg_explain.csv")