# OptSFC: Optimal Service Function Chaining


## :page_with_curl: What is OptSFC

OptSFC is a multi-objective reinforcement learning environment and training framework for optimal Moving Target Defense (MTD) strategies in Telco Cloud networks (*i.e.*, 5G and future 6G networks).

The environment simulates a small private 5G network architecture with a **core cluster** and an **edge cluster**, incorporating realistic cluster, VNF (Virtual Network Function), and CNF (Cloud-native Network Function) performance characteristics. Migration performances are simulated based on data collected from a real 5G lab testbed where these network functions were monitored during operation.

The threats simulated in the testbed are **CVE vulnerabilities** present in the VNFs/CNFs, quantified using **CVSS exploitability and base scores** to provide realistic security risk assessment and threat modeling.

### Deep-RL Action Space

For each VNF/CNF in the network, the deep-RL agent can select between two primary actions:

- **Migration**: Move the network function to a different cluster
- **Reinstantiation**: Recreate the network function in a new location

**CNF Migration** (Stateful): CNFs are stateful services, so their live migration involves actual transfer of the container service state, preserving active connections and data.

**VNF Migration** (Stateless): VNF migration is implemented as parallel reinstantiation with traffic redirection, where a new instance is created while gracefully redirecting traffic flows.


## :gear: Features

- **Multi-objective 5G network environment (MOfiveG_net)**
- **Choice between MOMDP and MDP modes**:
  - MOMDP (default): Multi-objective mode for MORL algorithms
  - MDP: Single-objective mode with hardcoded reward weights (0.4, 0.3, 0.3)
- **Flexible episode duration**:
  - Continuous training
  - Episodic training with daily, weekly, and monthly episodes (default: episodic with monthly episodes)
- **Support for various RL algorithms**: PPO, A2C, MaskablePPO
- **Multi-objective RL training**: EUPG and Envelope algorithms
- **Split training support**: Continue training models across multiple sessions (useful for federated learning)
- **Comprehensive evaluation and visualization tools**

## :package: Installation
Python 3.10 and Pip are required to install OptSFC.
Clone the repository. Then, within the repository run:
```bash
pip install -e .
```


## :rocket: Quick Start

### Deep RL Training (PPO Example)

```python
from optsfc import MOfiveG_net, train

# Create episodic environment for single-objective RL
env = MOfiveG_net("MlpPolicy", budget_reset="episodic", non_MORL=True) # episodic defaults to monthly episodes

# Create continuous environment where the MTD budget is reset "daily" or "weekly"
env = MOfiveG_net("MlpPolicy", budget_reset="daily", non_MORL=True)

# Train PPO agent
train(
    agent_type="PPO", # PPO, A2C, or MaskablePPO
    policy="MlpPolicy", 
    total_timesteps=100000,
    model_name="ppo_model",
    log_dir="./logs/ppo/",
    budget_reset="episodic"
)
```

### Multi-Objective RL Training (EUPG Example)

```python
from optsfc import MOfiveG_net, train_eupg

# Create environment for MORL (default)
env = MOfiveG_net("MlpPolicy", budget_reset="episodic")

# Train EUPG agent
train_eupg(
    total_timesteps=100000,
    model_name="eupg_model", 
    budget_reset="episodic"
)
```

### Split Training for Federated Learning

```python
from optsfc import split_train_eupg, split_train_Envelope

# Train EUPG in splits (useful for FL rounds)
split_train_eupg(
    total_timesteps=100000,
    timesteps_split=10000,  # Save model every 10k steps
    model_name="eupg_fl_model",
    budget_reset="episodic"
)

# Train Envelope in splits
split_train_Envelope(
    total_timesteps=100000,
    timesteps_split=10000,
    model_name="envelope_fl_model",
    budget_reset="episodic"
)
```

## :file_folder: Package Structure

- `optsfc.envs.mo_fiveg_mdp`: Core environment implementation
- `optsfc.envs.morl_train`: Training utilities for MORL algorithms  
- `optsfc.envs.short_simulated_testbed`: Testing and validation utilities
- `optsfc.envs.short_space_dict`: Environment space definitions

## :link: Dependencies

- gymnasium/gym: Environment interface
- stable-baselines3: RL algorithms
- sb3-contrib: Additional RL algorithms
- morl-baselines: Multi-objective RL algorithms
- torch: Deep learning framework
- numpy, matplotlib, scipy: Scientific computing

## :balance_scale: License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
