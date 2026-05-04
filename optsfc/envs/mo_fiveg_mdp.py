import contextlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
# GYM
import gymnasium as gym
from gym.spaces import Dict
from gymnasium.utils import seeding
from gymnasium import spaces

# local files
from .short_space_dict import (observation_dictionary, space_init, obs_init,
                                reward_init, update_agent_obs,
                                vnfs_size, cnfs_size)
from .short_simulated_testbed import (is_action_possible,
                                      get_new_simulated_observation,
                                      perform_action, get_rewards,
                                      one_step_duration,
                                      update_mtd_constraints,
                                      is_mtd_budget_zero,
                                      get_rewards_multiple_null_steps)
from .rdx import reward_difference_explanation, _build_log_entry
from optsfc.envs.ppo.critic import PPOQNet, PPOQTrainer
from optsfc.envs.eupg.decomposed_critic import DecomposedQNet, DecomposedQTrainer

import copy

# SB3
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecCheckNan, DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# MORL_baselines
from morl_baselines.multi_policy.envelope.envelope import Envelope
from optsfc.envs.eupg.eupg_explain import EUPG

rewards_coeff = [0.4, 0.3, 0.3]


def scalarization(reward: np.ndarray, weights=None) -> float:
    if reward.ndim == 1 and reward.size == 3:
        return float(
            reward[0] * rewards_coeff[0]
            + reward[1] * rewards_coeff[1]
            + reward[2] * rewards_coeff[2]
        )
    elif reward.ndim > 1 and reward.shape[1] == 3:
        res = float(
            sum(
                reward[:, 0] * rewards_coeff[0]
                + reward[:, 1] * rewards_coeff[1]
                + reward[:, 2] * rewards_coeff[2]
            )
        )
        print("res", res)
        return res


def dict_observation_to_array(observation):
    """Flatten the observation dict into a 1-D numpy array."""
    return np.hstack([arr.ravel() for arr in observation.values()])


def float_to_rgb_pixel(value):
    value     = 0 if value in (-np.inf, np.inf) else value
    min_value = -20
    max_value = 999999999
    norm_value   = (value - min_value) / (max_value - min_value)
    scaled_value = int(norm_value * (255 ** 3))
    r = scaled_value // (255 ** 2)
    g = (scaled_value % (255 ** 2)) // 255
    b = scaled_value % 255
    return [r, g, b]


def dict_observation_to_image(observation):
    """Convert the observation dict to a square RGB image for CNN policies."""
    obs_array = dict_observation_to_array(observation)
    obs_sqrt  = math.ceil(math.sqrt(obs_array.size))
    mult4 = False
    if obs_sqrt < 18:
        mult4    = True
        obs_sqrt *= 3

    obs_img = np.zeros(shape=(obs_sqrt, obs_sqrt, 3), dtype=np.uint8)
    for i in range(obs_sqrt):
        for j in range(obs_sqrt):
            if i * obs_sqrt + j >= obs_array.size:
                break
            if mult4:
                obs_img[i][j]     = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i + 1][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i][j + 1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i + 1][j + 1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i + 1][j + 2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i + 2][j + 1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i + 2][j + 2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i][j + 2]     = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i + 2][j]     = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                i += 2
                j += 2
            else:
                obs_img[i][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
    return obs_img


def sigmoid_schedule(t: float, p_start: float, p_end: float,
                     T: float, k: float = 4.0) -> float:
    """
    Logistic schedule for smoothly increasing a value from p_start to p_end
    over T steps.

    Parameters
    ----------
    t       : current time step
    p_start : initial parameter value
    p_end   : final parameter value
    T       : total steps over which the transition occurs
    k       : steepness of the sigmoid curve (default 4.0)
    """
    return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - T / 2) / T))


class MOfiveG_net(gym.Env):
    metadata = {"render.modes": ["console"]}

    # Temporary measurement attributes (can be removed without affecting logic).
    max_resource_penalty = 0
    max_network_penalty  = 0
    max_security_penalty = 0

    # MTD budget constraints per VNF per month.
    # Values are derived from a 99.99% SLA (0.05% downtime budget):
    #   migrations        : 379   (uses half the available downtime budget,
    #                              based on 330 ms per migration)
    #   reinstantiations  : 1798  (uses the other half)
    #   stateful_mig (CNF): 144   (uses the full 0.05% budget; CNFs support
    #                              only this MTD action; duration is taken
    #                              from simulated_testbed.py migration time)
    migrations_per_month            = 379
    reinstantiations_per_month      = 1798
    stateful_migrations_per_month   = 144
    # Monthly constraint reset interval in simulation steps.
    constraints_reset = 2592000 / one_step_duration

    # Maximum number of manageable network resources (VNFs + CNFs).
    max_resources = vnfs_size + cnfs_size
    # Action space: null action + 2 actions per VNF + 1 action per CNF.
    n_actions = 1 + vnfs_size * 2 + cnfs_size

    initial_recon_asp = 0.04
    recon_T = int(2592000 / one_step_duration)

    # Resource cost coefficients (unit: USD).
    # Formula: cost = intercept + coeff_cpu*cpu + coeff_ram*ram_gb + coeff_disk*disk_gb
    intercept  = -0.0820414
    coeff_cpu  =  0.03147484
    coeff_ram  =  0.00424486
    coeff_disk =  0.000066249

    # Attack type identifiers.
    RECON      = "recon"
    APT        = "apt"
    DOS        = "DoS"
    DATA_LEAK  = "data_leak"
    UNDEFINED  = "undefined"

    def __init__(self, policy, budget_reset="episodic",
                 non_MORL=False, rewards_coeff=rewards_coeff, num_envs=1):
        """
        Simplified 5G network environment with two edge domains and one core
        domain, used for MTD (Moving Target Defence) RL experiments.

        Parameters
        ----------
        policy       : str  "Cnn*", "Mlp*", or "MORL*"
        budget_reset : str  "episodic" | "weekly" | "daily"
        non_MORL     : bool  True → single-objective scalar reward;
                             False → 3-component MORL reward vector
        rewards_coeff: list[float]  scalarisation weights (must sum to 1)
        num_envs     : int  number of parallel environments (default 1)
        """
        self.explain_log = []

        # ── Temporal / episodic state trackers ────────────────────────────────
        # These two attributes are read by extract_state_features() in rdx.py
        # to compute "temporal pressure" features for the state-context RDX
        # visualisation (§7 of the evaluation script).
        #
        # last_mtd_step         : global step at which the most recent valid
        #                         MTD action (action != 0 and valid) was applied.
        #                         Reset to 0 at environment reset.
        # security_penalty_cumul: running sum of the per-step mean security
        #                         penalty across all active resources.  Provides
        #                         a measure of cumulative threat exposure over
        #                         the episode.  Reset to 0.0 at environment reset.
        self.last_mtd_step          = 0
        self.security_penalty_cumul = 0.0

        self.recon_schedule = [
            sigmoid_schedule(t=t, p_start=0.01, p_end=1, T=self.recon_T)
            for t in range(self.recon_T)
        ]

        # Scale monthly budgets to the chosen reset period.
        if budget_reset == "weekly":
            self.migrations_per_month           = self.migrations_per_month            / 30 * 7
            self.reinstantiations_per_month     = self.reinstantiations_per_month      / 30 * 7
            self.stateful_migrations_per_month  = self.stateful_migrations_per_month   / 30 * 7
            self.constraints_reset              = self.constraints_reset               / 30 * 7
        elif budget_reset == "daily":
            self.migrations_per_month           = self.migrations_per_month            / 30
            self.reinstantiations_per_month     = self.reinstantiations_per_month      / 30
            self.stateful_migrations_per_month  = self.stateful_migrations_per_month   / 30
            self.constraints_reset              = self.constraints_reset               / 30

        self.budget_reset  = budget_reset
        self.policy        = policy
        self.non_MORL      = non_MORL
        self.rewards_coeff = rewards_coeff

        if non_MORL:
            if len(rewards_coeff) != 3 and sum(rewards_coeff) != 1:
                raise ValueError(
                    "rewards_coeff must have 3 float values that sum to 1."
                )
            else:
                self.rewards_coeff = rewards_coeff

        self.environment      = copy.deepcopy(space_init(self))
        self.observation      = copy.deepcopy(obs_init(self))
        self.observation_space = Dict(observation_dictionary)
        self.step_counter     = 0

        if self.policy.startswith("Cnn"):
            image_shape = dict_observation_to_image(self.observation).shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=image_shape, dtype=np.uint8
            )
        else:   # Mlp or MORL
            flat_shape = dict_observation_to_array(self.observation).shape
            self.observation_space = spaces.Box(
                low=0, high=10000, shape=flat_shape, dtype=np.float16
            )

        self.reward_cumul             = 0
        self.constraints_reset_counter = 0
        self.action_space             = spaces.Discrete(self.n_actions)
        self.reward_space             = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,)
        )

        # Per-resource attack surface probability trackers.
        self.dynamic_asp = [
            {
                "dyn_recon_counter": 0,
                "recon":             self.initial_recon_asp,
                "apt":               0,
                "dos":               0,
                "data_leak":         0,
                "undefined":         0,
            }
            for _ in range(self.max_resources)
        ]
        self.reward_vector = copy.deepcopy(reward_init)

        if budget_reset == "episodic":
            self.spec = gym.envs.registration.EnvSpec(
                "MOfiveG_net", max_episode_steps=self.constraints_reset
            )
        else:
            self.spec = gym.envs.registration.EnvSpec("MOfiveG_net")

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state and return the first obs."""
        super().reset(seed=seed)
        self.non_MORL = self.non_MORL

        update_mtd_constraints(
            self.environment,
            self.migrations_per_month,
            self.reinstantiations_per_month,
            self.stateful_migrations_per_month,
        )
        self.constraints_reset_counter = 0
        self.reward_cumul              = 0
        self.reward_noScalar           = 0
        self.constraints_reset_counter = 0

        self.environment   = copy.deepcopy(space_init(self))
        self.observation   = copy.deepcopy(obs_init(self))
        self.reward_vector = copy.deepcopy(reward_init)

        # Reset per-resource attack surface probability trackers.
        self.dynamic_asp = [
            {
                "dyn_recon_counter": 0,
                "recon":             self.initial_recon_asp,
                "apt":               0,
                "dos":               0,
                "data_leak":         0,
                "undefined":         0,
            }
            for _ in range(self.max_resources)
        ]

        # Reset temporal / episodic state trackers used by extract_state_features().
        self.last_mtd_step          = 0
        self.security_penalty_cumul = 0.0

        if self.policy.startswith("Cnn"):
            return dict_observation_to_image(self.observation), {}
        else:
            return dict_observation_to_array(self.observation), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        info = {}
        self.step_counter += 1

        # Snapshot the observation *before* the action is applied.
        # This array is used as input to the Q-network inside RDX and is also
        # passed to extract_state_features() so that Q-value differences and
        # state features are semantically aligned (same s_t).
        obs_before_step = dict_observation_to_array(self.observation)

        # Increment the dynamic reconnaissance counter for each resource.
        for i in range(self.max_resources):
            self.dynamic_asp[i]["dyn_recon_counter"] += 1

        truncated          = False
        to_reset           = False
        depleted_mtd_budget = False

        self.constraints_reset_counter += 1
        if self.budget_reset != "episodic" and \
                self.constraints_reset_counter >= self.constraints_reset:
            update_mtd_constraints(
                self.environment,
                self.migrations_per_month,
                self.reinstantiations_per_month,
                self.stateful_migrations_per_month,
            )
            self.constraints_reset_counter = 0
        elif self.budget_reset == "episodic" and \
                self.constraints_reset_counter >= self.constraints_reset:
            to_reset              = True
            truncated             = True
            info["jumped_steps"]  = 0

        if self.budget_reset == "episodic" and is_mtd_budget_zero(self.environment):
            to_reset            = True
            depleted_mtd_budget = True

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        valid_action, err = is_action_possible(self.environment, action)

        self.reward_vector = copy.deepcopy(reward_init)

        if not valid_action:
            final_reward = -20
        elif action == 0:   # null action: no MTD applied
            final_reward = -10
        else:
            final_reward = 0
            perform_action(self, self.environment, action, self.reward_vector)

            # Track the most recent step at which a valid MTD action was applied.
            # Used by extract_state_features() to compute "steps_since_last_mtd",
            # which captures temporal pressure (how long ago the agent last acted).
            self.last_mtd_step = self.step_counter

        get_new_simulated_observation(self.environment)
        get_rewards(self, self.environment, self.reward_vector)

        if self.budget_reset == "episodic" and to_reset and depleted_mtd_budget:
            remaining_steps     = self.constraints_reset - self.constraints_reset_counter
            info["jumped_steps"] = remaining_steps
            for i in range(min(100, int(remaining_steps))):
                get_new_simulated_observation(self.environment)
                get_rewards(self, self.environment, self.reward_vector)
            remaining_steps -= min(100, int(remaining_steps))
            if remaining_steps > 0:
                get_rewards_multiple_null_steps(
                    self, self.environment, self.reward_vector, remaining_steps
                )

        self.reward_noScalar = [
            self.reward_vector["resource_reward"]            + final_reward,
            self.reward_vector["network_reward"]             + final_reward,
            self.reward_vector["proactive_security_reward"]  + final_reward,
        ]

        final_reward += float(
            self.reward_vector["resource_reward"]           * self.rewards_coeff[0]
            + self.reward_vector["network_reward"]          * self.rewards_coeff[1]
            + self.reward_vector["proactive_security_reward"] * self.rewards_coeff[2]
        )

        self.reward_cumul += final_reward

        # Update the cumulative security penalty tracker.
        # Accumulates the mean per-step security penalty across all active
        # resources; used by extract_state_features() as a measure of
        # cumulative threat exposure within the episode.
        nb_res = int(self.environment["nb_resources"][0])
        if nb_res > 0:
            self.security_penalty_cumul += float(
                self.environment["security_penalty"][:nb_res].mean()
            )

        done = self.reward_cumul <= -100000000 or to_reset

        info["rew"]      = final_reward
        self.observation = update_agent_obs(self.environment, self.observation)

        # ── Explainability ────────────────────────────────────────────────────
        if getattr(self, "model_for_explain", None) is not None:
            try:
                explanation = reward_difference_explanation(
                    self.model_for_explain,
                    obs_before_step,
                    weights=self.rewards_coeff,
                    env_action=action,
                    env=self,
                )
                info["explanation"] = explanation
                self.explain_log.append(
                    _build_log_entry(
                        self.step_counter,
                        action,
                        explanation,
                        # Pass obs_before_step and self so that _build_log_entry
                        # can call extract_state_features() and append "feat_*"
                        # columns to every CSV row.  obs_before_step is the same
                        # array fed into the Q-network, ensuring semantic alignment
                        # between Q-value differences and state context features.
                        obs_array=obs_before_step,
                        env=self,
                    )
                )
            except Exception as e:
                info["explanation_error"] = str(e)
                print(f"❌ step {self.step_counter}: {e}")
                import traceback; traceback.print_exc()

        # ── Auxiliary critic update (PPO / A2C / EUPG) ────────────────────────
        if getattr(self, "critic_trainer", None) is not None:
            if isinstance(self.critic_trainer, DecomposedQTrainer):
                # EUPG: store per-objective reward vector.
                self.critic_trainer.store(
                    obs_before_step,
                    action,
                    np.array(self.reward_noScalar, dtype=np.float32),
                    dict_observation_to_array(self.observation),
                    done,
                )
            else:
                # PPO / A2C: store scalar reward.
                self.critic_trainer.store(
                    obs_before_step,
                    action,
                    float(info["rew"]),
                    dict_observation_to_array(self.observation),
                    done,
                )
            self.critic_trainer.update()

        if self.non_MORL:
            if self.policy.startswith("Cnn"):
                return (dict_observation_to_image(self.observation),
                        final_reward, done, truncated, info)
            elif self.policy.startswith("Mlp"):
                return (dict_observation_to_array(self.observation),
                        final_reward, done, truncated, info)
        else:   # MORL
            return (dict_observation_to_array(self.observation),
                    np.array(self.reward_noScalar), done, truncated, info)

    # ── Utility ───────────────────────────────────────────────────────────────

    def save_explanations(self, filename="explanations.csv"):
        """Persist the accumulated explanation log to a CSV file."""
        pd.DataFrame(self.explain_log).to_csv(filename, index=False)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="console"):
        pass

    def close(self):
        pass

    # ── Dynamic action mask ───────────────────────────────────────────────────

    def dyn_action_mask(self, action_num):
        """
        Return True if action_num is currently feasible.

        Rules:
          action 0   : always valid (null action)
          action > 0 : target VNF/CNF must not already be under MTD, must have
                       remaining budget, and the host VIM must have enough
                       free resources.
        """
        if action_num > (self.environment["nb_resources"][0] * 2):
            return False
        if action_num == 0:
            return True

        vnf_index = int((action_num - 1) / 2)

        # Check remaining MTD budget (restart or migrate share the same counter).
        if self.environment["mtd_constraint"][vnf_index][0] == 0:
            return False

        # Block if the resource is already undergoing an MTD operation.
        if self.environment["mtd_action"][vnf_index][0] != 0:
            return False

        # Block if the host VIM has insufficient free resources.
        vim_idx = self.environment["vim_host"][vnf_index][0]
        if (
            self.environment["resource_consumption"][vnf_index][0]
            < self.environment["vim_resources"][vim_idx][0]
            and self.environment["resource_consumption"][vnf_index][1]
            < self.environment["vim_resources"][vim_idx][1]
            and self.environment["resource_consumption"][vnf_index][2]
            < self.environment["vim_resources"][vim_idx][2]
        ):
            return True
        return False

    def action_masks(self) -> list:
        return [self.dyn_action_mask(a) for a in range(self.n_actions)]


# ── Training callback ─────────────────────────────────────────────────────────

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Save the model whenever the mean training reward improves (checked every
    check_freq steps) and periodically every 100k steps.
    """
    def __init__(self, check_freq: int, log_dir: str, model_name: str,
                 policy: str, env: Monitor, verbose=False):
        super().__init__(verbose)
        self.check_freq       = check_freq
        self.log_dir          = log_dir
        self.save_path        = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.model_name       = model_name
        self.policy           = policy
        self.env              = env
        self.prev_rew         = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            cumul_reward = sum(self.env.episode_returns) + sum(self.env.rewards)
            ep_rew  = sum(self.env.rewards)
            ep_len  = len(self.env.rewards)
            ep_info = {
                "r": round(ep_rew - self.prev_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.env.t_start, 6),
            }
            if self.env.results_writer:
                self.env.results_writer.write_row(ep_info)
            self.prev_rew = ep_rew

            time.sleep(1)
            try:
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(
                            f"Best mean reward: {self.best_mean_reward:.2f} "
                            f"- Last mean reward: {mean_reward:.2f}"
                        )
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose:
                            print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)
            except Exception as e:
                print("Error in SaveOnBestTrainingRewardCallback:", e)

            if self.env.total_steps % 100000 == 0:
                self.model.save(
                    self.save_path + "_" + str(self.env.total_steps)
                )

            self.logger.record("reward/return", cumul_reward)
            self.logger.record(
                "reward/mean_reward",
                cumul_reward / (0.001 + self.env.total_steps),
            )
        return True


# ── Scalar algorithm training ─────────────────────────────────────────────────

def train(agent_type, policy, total_timesteps, model_name,
          log_dir, budget_reset="episodic"):
    """
    Train a single-objective RL agent (DQN, A2C, PPO, DDPG, SAC, TD3, or
    MaskablePPO) with the RDX explainability hook enabled.

    The explanation log is written to <log_dir>/<model_name>_explain.csv after
    training completes.
    """
    os.makedirs(log_dir, exist_ok=True)
    env_train = Monitor(
        MOfiveG_net(policy, budget_reset=budget_reset, non_MORL=True), log_dir
    )
    env_train.action_space.seed(123)

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=5000, log_dir=log_dir,
        model_name=model_name, policy=policy, env=env_train,
    )

    print(f"Start training the {agent_type} agent")
    if agent_type == "DQN":
        model = DQN(policy, env_train, verbose=1,
                    tensorboard_log=f"./tmp/{model_name}/")
    elif agent_type == "A2C":
        model = A2C(policy, env_train, verbose=1,
                    tensorboard_log=f"./tmp/{model_name}/")
    elif agent_type == "PPO":
        model = PPO(policy, env_train, verbose=1,
                    tensorboard_log=f"./tmp/{model_name}/")
    elif agent_type == "DDPG":
        model = DDPG(policy, env_train, verbose=1,
                     tensorboard_log=f"./tmp/{model_name}/")
    elif agent_type == "SAC":
        model = SAC(policy, env_train, verbose=1,
                    tensorboard_log=f"./tmp/{model_name}/")
    elif agent_type == "TD3":
        model = TD3(policy, env_train, verbose=1,
                    tensorboard_log=f"./tmp/{model_name}/")
    else:   # MaskablePPO
        model = MaskablePPO(policy, env_train, verbose=1,
                            tensorboard_log=f"./tmp/{model_name}/")

    env_train.env.model_for_explain = model

    obs_dim = env_train.env.observation_space.shape[0]

    if agent_type == "PPO":
        ppo_q         = PPOQNet(obs_dim=obs_dim, n_actions=env_train.env.n_actions)
        ppo_q_trainer = PPOQTrainer(ppo_q, gamma=0.99)
        model.ppo_q_net              = ppo_q
        env_train.env.critic_trainer = ppo_q_trainer

    elif agent_type == "A2C":
        a2c_q         = PPOQNet(obs_dim=obs_dim, n_actions=env_train.env.n_actions)
        a2c_q_trainer = PPOQTrainer(a2c_q, gamma=0.99)
        model.a2c_q_net              = a2c_q
        env_train.env.critic_trainer = a2c_q_trainer

    with open(log_dir + "Log" + model_name + ".txt", "a") as f:
        with contextlib.redirect_stdout(f):
            model.learn(total_timesteps, callback=callback)

    plot_results(log_dir, f"OptSFC {agent_type} Learning Curve").savefig(
        log_dir + "plot_" + model_name + ".pdf"
    )
    model.save(log_dir + model_name + "last")

    if env_train.env.explain_log:
        df_log     = pd.DataFrame(env_train.env.explain_log)
        match_rate = df_log["match"].mean() * 100
        print(f"Explanation match rate: {match_rate:.1f}%")
        print(f"   Matched: {df_log['match'].sum()} / {len(df_log)}")

    actual_env = env_train.env
    del model
    gc.collect()
    return actual_env


# ── MORL training helpers ─────────────────────────────────────────────────────

def train_eupg(total_timesteps, model_name, budget_reset="episodic"):
    """
    Train an EUPG agent with the DecomposedQNet RDX explainability hook.

    The explanation log is written to <model_name>_explain.csv after training.
    """
    env      = MOfiveG_net("MlpPolicy", budget_reset)
    eval_env = MOfiveG_net("MlpPolicy", budget_reset)
    save_dir = "models"

    weights = np.array(rewards_coeff)
    agent   = EUPG(env, scalarization=scalarization, weights=weights,
                   gamma=0.99, log=False, learning_rate=0.001)

    obs_dim = env.observation_space.shape[0]
    q_net   = DecomposedQNet(obs_dim=obs_dim, n_actions=env.n_actions)
    trainer = DecomposedQTrainer(q_net, weights=rewards_coeff, lr=3e-4)

    agent.decomposed_q_net  = q_net
    env.model_for_explain   = agent
    env.critic_trainer      = trainer

    agent.train(total_timesteps=total_timesteps, eval_env=eval_env)

    if env.explain_log:
        df_log     = pd.DataFrame(env.explain_log)
        match_rate = df_log["match"].mean() * 100
        print(f"Explanation match rate: {match_rate:.1f}%")
        print(f"   Matched: {df_log['match'].sum()} / {len(df_log)}")

    return env


# ── Plotting helpers ──────────────────────────────────────────────────────────

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """Plot per-episode reward from a Monitor log directory."""
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = np.diff(y)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title)
    return plt


# ── Federated-learning helpers ────────────────────────────────────────────────

def initialize_model_for_flwr(rl_algo, log_dir, budget_reset):
    policy     = "MlpPolicy"
    train_dir  = log_dir
    eval_dir   = "./tested_models/"

    if rl_algo in ["PPO", "A2C", "MaskablePPO"]:
        env_train = Monitor(
            MOfiveG_net("MlpPolicy", budget_reset, non_MORL=True), train_dir
        )
        eval_env = MOfiveG_net("MlpPolicy", budget_reset, non_MORL=True)
    else:
        env_train = MOfiveG_net("MlpPolicy", budget_reset)
        eval_env  = MOfiveG_net("MlpPolicy", budget_reset)

    env_train.action_space.seed(123)
    eval_env.action_space.seed(123)

    if rl_algo == "Envelope":
        model = Envelope(
            env_train,
            learning_rate=3e-4, gamma=0.99,
            initial_epsilon=0.01, final_epsilon=0.01,
            batch_size=256, net_arch=[256, 256, 256, 256], log=False,
        )
    elif rl_algo == "EUPG":
        model = EUPG(env_train, scalarization=scalarization,
                     weights=np.array(rewards_coeff),
                     gamma=0.99, log=False, learning_rate=0.001)
    elif rl_algo == "A2C":
        model = A2C(policy, env_train)
    elif rl_algo == "PPO":
        model = PPO(policy, env_train)
    elif rl_algo == "MaskablePPO":
        model = MaskablePPO(policy, env_train)

    return model, env_train, eval_env


def train_in_flwr(model, train_env, total_timesteps, model_name, log_dir):
    policy     = "MlpPolicy"
    train_dir  = "./trained_models/"
    model_name = "PPO_model"

    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=5000, log_dir=train_dir,
        model_name=model_name, policy=policy, env=train_env,
    )
    with open(log_dir + "Log" + model_name + ".txt", "a") as f:
        with contextlib.redirect_stdout(f):
            model.learn(total_timesteps, callback=callback)

    plot_results(log_dir, f"OptSFC {policy} Learning Curve").savefig(
        log_dir + "plot_" + model_name + ".pdf"
    )
    return model, train_env


# ── Gym registration ──────────────────────────────────────────────────────────

from gymnasium.envs.registration import register

register(
    id="MOfiveG_net-v0",
    entry_point="short_episodic_mo_fiveg_mdp:MOfiveG_net",
)