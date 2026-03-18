from optsfc import MOfiveG_net, train_Envelope
import torch

# Train Envelope agent
env = train_Envelope(
    total_timesteps=5000,
    model_name="envelope_model",
    budget_reset="episodic"
)

env.save_explanations("envelope_explain.csv")