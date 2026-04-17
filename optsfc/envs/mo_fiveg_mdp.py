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

#local files
from .short_space_dict import observation_dictionary, space_init, obs_init, reward_init, update_agent_obs, vnfs_size, cnfs_size
from .short_simulated_testbed import is_action_possible, get_new_simulated_observation, perform_action, get_rewards, one_step_duration, update_mtd_constraints, is_mtd_budget_zero, get_rewards_multiple_null_steps
from .rdx import reward_difference_explanation, _build_log_entry
from optsfc.envs.ppo.critic import PPOQNet, PPOQTrainer
from optsfc.envs.eupg.decomposed_critic import DecomposedQNet, DecomposedQTrainer

import copy

# SB3
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from sb3_contrib.common.maskable.utils import get_action_masks
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecCheckNan, DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# MORL_baselines
from morl_baselines.multi_policy.envelope.envelope import Envelope
# from morl_baselines.single_policy.esr.eupg import EUPG
from optsfc.envs.eupg.eupg_explain import EUPG

rewards_coeff = [0.4, 0.3, 0.3]

def scalarization(reward: np.ndarray, weights= None) -> float:
    if reward.ndim == 1 and reward.size == 3:
        return float(reward[0] * rewards_coeff[0] + reward[1] * rewards_coeff[1] + reward[2] * rewards_coeff[2])
    elif reward.ndim > 1 and reward.shape[1] == 3:
        res = float(sum(reward[:, 0] * rewards_coeff[0] + reward[:, 1] * rewards_coeff[1] + reward[:, 2] * rewards_coeff[2]))
        print("res", res)
        return res
        
        
def dict_observation_to_array(observation):
    # Convert the dictionary observation into a numpy array
    obs_array = np.hstack([arr.ravel() for arr in observation.values()])
    return obs_array


def float_to_rgb_pixel(value):
    value = 0 if value == -np.inf or value == np.inf else value
    min_value = -20
    max_value = 999999999
    norm_value = (value - min_value) / (max_value - min_value)
    # Scale the value to the range [0, 255^3]
    scaled_value = int(norm_value * (255 ** 3))
    # Divide the value into three 8-bit integers
    r = scaled_value // (255 ** 2)
    g = (scaled_value % (255 ** 2)) // 255
    b = scaled_value % 255
    # Return the RGB pixel
    return [r, g, b]


def dict_observation_to_image(observation):
    # Convert the dictionary observation into a numpy array
    obs_array = dict_observation_to_array(observation)
    # give the sqrt of obs_array.size
    obs_sqrt = math.ceil(math.sqrt(obs_array.size))
    # for DQN_CNN this is needed to ensure the image is equal or bigger than the kernel
    mult4 = False
    if obs_sqrt < 18:
        mult4 = True
        obs_sqrt *= 3
    # print("the array size is",obs_sqrt,"x",obs_sqrt)

    obs_img = np.zeros(shape=(obs_sqrt, obs_sqrt, 3), dtype=np.uint8)
    for i in range(obs_sqrt):
        for j in range(obs_sqrt):
            if i * obs_sqrt + j >= obs_array.size:
                break
            #increase image by 4 if needed
            if mult4:
                obs_img[i][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+1][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i][j+1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+1][j+1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+1][j+2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+2][j+1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+2][j+2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i][j+2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+2][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                i += 2
                j += 2
            else:
                obs_img[i][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
    # obs_img = Image.fromarray(obs_img)
    return obs_img


def sigmoid_schedule(t: float, p_start: float, p_end: float, T: float, k: float = 4.0) -> float:
    """
    Sigmoid/logistic schedule for increasing a value over time.

    Args:
        t (float): Current time step (e.g., seconds or steps).
        p_start (float): Initial value of the parameter.
        p_end (float): Final value of the parameter.
        T (float): Total time/steps over which to increase.
        k (float): Steepness of the sigmoid curve (default: 10.0).

    Returns:
        float: Adjusted parameter value at time `t`.
    """
    return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - T / 2) / T))


class MOfiveG_net(gym.Env):
    metadata = {'render.modes': ['console']}
    # temporary attributes for mesurements (can be removed)
    max_resource_penalty = 0
    max_network_penalty = 0
    max_security_penalty = 0

    # MTD stateless constraints per month per vnf (based on TopoFuzzer paper but with SLA of 99.99, and not 99.999%)
    # these values are NOT derived from the migration time in simulated_testbed.py but from the downtime of the vnf (330 ms)
    migrations_per_month = 379 # this takes half of the available 0.05% of the SLA
    reinstantiations_per_month = 1798 # this takes the other half of the available 0.05% of the SLA
    # next value with SLA 99.95%. This value takes full 0.05% budget as CNFs only have this action possible. The budget IS derived from the migration time in simulated_testbed.py
    stateful_migrations_per_month = 144
    constraints_reset = 2592000 / one_step_duration # reset monthly constraints every month (2,592,000 = seconds in a month)

    # MTD stateful container constraints per month

    # (HYPERPARAMETER) MAX number of network resources *VDUs manageable in the environment
    max_resources = vnfs_size + cnfs_size
    # n_actions = null action + 2 actions per vnf + 1 action per cnf
    n_actions = 1 + vnfs_size * 2 + cnfs_size # dynamic n_actions count
    initial_recon_asp = 0.04
    recon_T = int(2592000 / one_step_duration)

    # measure the resouce cost of the operation in near real-time and aggregate it to previous results of the same tuple (action, resource_type / size_unit) to have a better mean
    #       the resource cost unit is $, determined by formula $=intercept + coeffcpu * cpu + coeffram * ram_gb + coeffdisk * disk_gb
    intercept = -0.0820414
    coeff_cpu = 0.03147484
    coeff_ram = 0.00424486
    coeff_disk= 0.000066249

    # attack types
    RECON = 'recon' # recon asp increase at every step with recon_asp_factor
    APT = 'apt' # apt_asp depends on cves and recon_asp
    DOS = 'DoS' # dos_asp depends on cves and recon_asp
    DATA_LEAK = 'data_leak' # data_leak_asp depends on cves and recon_asp
    UNDEFINED = 'undefined'

    ''''
    For PROACTIVE part: evaluate attack surface of each resource based on CVEs and the CVSS scores.
     Use CVSS exploitability in real-time and change it based on MTD actions:
     1) if vulnerability needs IP and the MTD action applied changes it than reduce ASP
     2) if vulnerability needs port //             //                //
     3) if vulnerability needs OS   //             //                //
     4) if attack source is blacklisted
     5)
    '''

    def __init__(self, policy, budget_reset="episodic", non_MORL=False, rewards_coeff=rewards_coeff, num_envs=1):
        """
        This Gym environment is a simplified version of a 5G testbed with two edge domains and a core domain.
        :param policy: The policy to use for the agent. Can be "Cnn", "Mlp", or "MORL"
        :param budget_reset: "episodic" or "weekly" or "daily"
        :param non_MORL: if True, the environment is used for a single objective RL agent, if False, the environment is used for MORL
        :param rewards_coeff: if non_MORL is True, this is the reward coefficients for the three objectives
        :param num_envs: default is 1
        """
        self.explain_log = []

        self.recon_schedule = [sigmoid_schedule(t=t, p_start=0.01, p_end=1, T=self.recon_T) for t in range(self.recon_T)]
        # change these parameters based on wether they are weekly or daily
        if budget_reset == "weekly":
            self.migrations_per_month = self.migrations_per_month / 30 * 7
            self.reinstantiations_per_month = self.reinstantiations_per_month / 30 * 7
            self.stateful_migrations_per_month = self.stateful_migrations_per_month / 30 * 7
            self.constraints_reset = self.constraints_reset / 30 * 7
        elif budget_reset == "daily":
            self.migrations_per_month = self.migrations_per_month / 30
            self.reinstantiations_per_month = self.reinstantiations_per_month / 30
            self.stateful_migrations_per_month = self.stateful_migrations_per_month / 30
            self.constraints_reset = self.constraints_reset / 30
        self.budget_reset = budget_reset
        self.policy = policy
        self.non_MORL = non_MORL
        self.rewards_coeff = rewards_coeff
        if non_MORL:
            if len(rewards_coeff) != 3 and sum(rewards_coeff) != 1:
                raise ValueError("rewards_coeff parameter of MOfiveG_net must have 3 float values, the sum of which is 1.")
            else:
                self.rewards_coeff = rewards_coeff
        self.environment = copy.deepcopy(space_init(self))
        self.observation = copy.deepcopy(obs_init(self))
        self.observation_space = Dict(observation_dictionary)
        self.step_counter = 0

        if self.policy.startswith("Cnn"):
            image_shape = dict_observation_to_image(self.observation).shape
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape,
                                                dtype=np.uint8)
        else:# "Mlp" or "MORL"
            flat_observation_shape = dict_observation_to_array(self.observation).shape
            self.observation_space = spaces.Box(low=0, high=10000, shape=flat_observation_shape,
                                                   dtype=np.float16)
        self.reward_cumul = 0
        self.constraints_reset_counter = 0
        self.action_space = spaces.Discrete(self.n_actions)
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        # init observation to zero for all features
        self.dynamic_asp = [{'dyn_recon_counter': 0, 'recon': self.initial_recon_asp, 'apt': 0, 'dos': 0, 'data_leak': 0, 'undefined': 0} for _ in range(self.max_resources)]
        self.reward_vector = copy.deepcopy(reward_init)

        # create a gym spec for this environment
        if budget_reset == "episodic":
            self.spec = gym.envs.registration.EnvSpec("MOfiveG_net", max_episode_steps= self.constraints_reset)
        else:
            self.spec = gym.envs.registration.EnvSpec("MOfiveG_net")

    def reset(self, seed = None, options = None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed)
        self.non_MORL = self.non_MORL
        # reset mtd constraints
        update_mtd_constraints(self.environment, self.migrations_per_month, self.reinstantiations_per_month, self.stateful_migrations_per_month)
        self.constraints_reset_counter = 0
        # reset counters
        self.reward_cumul = 0
        self.reward_noScalar = 0
        self.constraints_reset_counter = 0
        # init observation
        self.environment = copy.deepcopy(space_init(self))
        self.observation = copy.deepcopy(obs_init(self))
        self.reward_vector = copy.deepcopy(reward_init)

        # reset asp vector
        self.dynamic_asp = [{'dyn_recon_counter': 0, 'recon': self.initial_recon_asp, 'apt': 0, 'dos': 0, 'data_leak': 0, 'undefined': 0} for _ in range(self.max_resources)]
        if self.policy.startswith("Cnn"):
            return dict_observation_to_image(self.observation), {}
        else:# "Mlp" or "MORL"
            return dict_observation_to_array(self.observation), {}
    
    def save_explanations(self, filename="explanations.csv"):
        df = pd.DataFrame(self.explain_log)
        df.to_csv(filename, index=False)

    def step(self, action):
        info = {}
        # increment steps counter
        self.step_counter += 1

        obs_before_step = dict_observation_to_array(self.observation)

        # increment dynamic recon counter for each vnf
        for i in range(self.max_resources):
            self.dynamic_asp[i]['dyn_recon_counter'] += 1
        # print("step number ", self.step_counter)

        # add truncated for new Gym API
        truncated = False
        to_reset = False
        depleted_mtd_budget = False
        # if episodic and budget is depleted then reset the episode
        self.constraints_reset_counter += 1
        if self.budget_reset != "episodic" and self.constraints_reset_counter >= self.constraints_reset:
            update_mtd_constraints(self.environment, self.migrations_per_month, self.reinstantiations_per_month, self.stateful_migrations_per_month)
            self.constraints_reset_counter = 0
        elif self.budget_reset == "episodic" and self.constraints_reset_counter >= self.constraints_reset:
            to_reset = True
            truncated = True
            info["jumped_steps"] = 0
        if self.budget_reset == "episodic" and is_mtd_budget_zero(self.environment):
            to_reset = True
            depleted_mtd_budget = True


        # DYNAMIC ACTION-SPACE UPDATE
        # self.action_space = spaces.Discrete(self.environment['nb_resources'][0] * 2 + 1)

        # message error if the action is invalid
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        valid_action, err = is_action_possible(self.environment, action)

        # initialize step reward
        self.reward_vector = copy.deepcopy(reward_init)

        if not valid_action:
            final_reward = -20
            # print('action not performed', err)
        elif action == 0: # null action
            final_reward = -10
        else:
            final_reward = 0
            # update network state based on RL agent action
            perform_action(self, self.environment, action, self.reward_vector)

        # final_reward = 0

        # simulate network and update observation
        get_new_simulated_observation(self.environment)
        # print("actions array is ", self.environment['mtd_action'])
        # print("action constraints array is ", self.environment['mtd_constraint'])

        get_rewards(self, self.environment, self.reward_vector)

        # the bottom three variables are for LOG PURPOSES ONLY, CAN BE REMOVED
        # self.max_resource_penalty = max(self.max_resource_penalty, abs(reward_vector['resource_reward']))
        # self.max_network_penalty = max(self.max_network_penalty, abs(reward_vector['network_reward']))
        # self.max_security_penalty = max(self.max_security_penalty, abs(reward_vector['proactive_security_reward']))
        # print('resource_reward: ', self.reward_vector['resource_reward'], 'network_reward: ', self.reward_vector['network_reward'], 'proactive_security_reward', self.reward_vector['proactive_security_reward'])
        # get the scaled unified reward
        # print('resource_reward: ', self.reward_vector['resource_reward'], 'network_reward: ', self.reward_vector['network_reward'], 'proactive_security_reward', self.reward_vector['proactive_security_reward'])
        # print('the action is', action)
        # time.sleep(0.2)

        if self.budget_reset == "episodic" and to_reset and depleted_mtd_budget:
            remaining_steps = self.constraints_reset - self.constraints_reset_counter
            info["jumped_steps"] = remaining_steps
            for i in range(min(100, int(remaining_steps))):
                get_new_simulated_observation(self.environment)
                get_rewards(self, self.environment, self.reward_vector)
            # get the last rewards and multiply it by the remaining steps to complete a month
            remaining_steps = remaining_steps - min(100, int(remaining_steps))
            if remaining_steps > 0:
                get_rewards_multiple_null_steps(self, self.environment, self.reward_vector, remaining_steps)

        # print("the reward vector is ", self.reward_vector)
        # multi-objective reward: share the penalty in final_reward between the three objectives
        self.reward_noScalar = [self.reward_vector['resource_reward'] + final_reward,
                                self.reward_vector['network_reward'] + final_reward,
                                self.reward_vector['proactive_security_reward'] + final_reward]

        # mono objective scalarized reward
        final_reward += float(self.reward_vector['resource_reward'] * self.rewards_coeff[0] + self.reward_vector['network_reward'] * self.rewards_coeff[1] + self.reward_vector['proactive_security_reward'] * self.rewards_coeff[2])

        self.reward_cumul += final_reward

        # game is done when we reach max number of episodes or reward is really bad
        if self.reward_cumul <= -100000000 or to_reset:
            done = True
        else:
            done = False

        info["rew"] = final_reward
        self.observation = update_agent_obs(self.environment, self.observation)

        # === Explainability ===
        if getattr(self, "model_for_explain", None) is not None:
            try:
                explanation = reward_difference_explanation(
                    self.model_for_explain,
                    obs_before_step,           
                    weights=self.rewards_coeff,
                    env_action=action,
                    env=self
                )
                info["explanation"] = explanation
                self.explain_log.append(
                    _build_log_entry(
                        self.step_counter, action, explanation
                    )
                )
            # except Exception as e:
            #     info["explanation_error"] = str(e)
            except Exception as e:
                info["explanation_error"] = str(e)
                print(f"❌ step {self.step_counter}: {e}")
                import traceback; traceback.print_exc()
        
        if getattr(self, "critic_trainer", None) is not None:
            if isinstance(self.critic_trainer, DecomposedQTrainer):
                # EUPG: per-objective reward vector
                self.critic_trainer.store(
                    obs_before_step,
                    action,
                    np.array(self.reward_noScalar, dtype=np.float32),
                    dict_observation_to_array(self.observation),
                    done
                )
            else:
                # PPO / A2C: scalar reward
                self.critic_trainer.store(
                    obs_before_step,
                    action,
                    float(info["rew"]),
                    dict_observation_to_array(self.observation),
                    done
                )
            self.critic_trainer.update()

        if self.non_MORL:
            if self.policy.startswith("Cnn"):
                return dict_observation_to_image(self.observation), final_reward, done, truncated, info
            elif self.policy.startswith("Mlp"):
                return dict_observation_to_array(self.observation), final_reward, done, truncated, info
        else: # MORL
            return dict_observation_to_array(self.observation), np.array(self.reward_noScalar), done, truncated, info


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='console'):
        # if self.reward_cumul % 100:
        #     print('the cumul reward is ', self.reward_cumul)
        pass

    def close(self):
        pass


    # DYNAMIC ACTION SPACE MASK
    def dyn_action_mask(self, action_num):
        # on every vnf 2 actions can be applied
        if action_num > (self.environment['nb_resources'][0] * 2):
            return False

        if action_num == 0:
            return True

        # for the vnf targetted check that it is not under MTD already, that the limit of MTD is not reached and that the amount of cpu, ram and disk needed are available
        vnf_index = int((action_num-1)/2)

        # check that the limit of MTDs possible is not reached
        if (action_num - 1) % 2 == 0:
            # action is a restart
            if self.environment['mtd_constraint'][vnf_index][0] == 0:
                return False
        else:
            # action is a migrate
            if self.environment['mtd_constraint'][vnf_index][0] == 0:
                return False

        if self.environment['mtd_action'][vnf_index][0] != 0:
            return False
        if self.environment['resource_consumption'][vnf_index][0] < \
                self.environment['vim_resources'][self.environment['vim_host'][vnf_index][0]][0] and \
                self.environment['resource_consumption'][vnf_index][1] < \
                self.environment['vim_resources'][self.environment['vim_host'][vnf_index][0]][1] and \
                self.environment['resource_consumption'][vnf_index][2] < \
                self.environment['vim_resources'][self.environment['vim_host'][vnf_index][0]][2]:
            return True
        else:
            return False


    def action_masks(self) -> [bool]:
        bools = []
        for action in range(0, self.n_actions):
            bools.append(self.dyn_action_mask(action))
        return bools


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, model_name: str, policy: str, env: Monitor, verbose=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.model_name = model_name
        self.policy = policy
        self.env = env
        self.prev_rew = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            cumul_reward = sum(self.env.episode_returns) + sum(self.env.rewards)
            ep_rew = sum(self.env.rewards)
            ep_len = len(self.env.rewards)
            ep_info = {"r": round(ep_rew - self.prev_rew, 6), "l": ep_len, "t": round(time.time() - self.env.t_start, 6)}
            if self.env.results_writer:
                self.env.results_writer.write_row(ep_info)
            self.prev_rew = ep_rew

            # Retrieve training reward
            time.sleep(1)
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                  # Mean training reward over the last 100 episodes
                  mean_reward = np.mean(y[-100:])
                  if self.verbose:
                      print("Num timesteps: {}".format(self.num_timesteps))
                      print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                  # New best model, you could save the agent here
                  if mean_reward > self.best_mean_reward:
                      self.best_mean_reward = mean_reward
                      # saving best model
                      if self.verbose:
                        print("Saving new best model to {}".format(self.save_path))
                      self.model.save(self.save_path)
            except Exception as e:
                print("Error in SaveOnBestTrainingRewardCallback: ", e)
                # continue the training

            # save the model if env.step_counter is a multiple of 333000
            if self.env.total_steps % 100000 == 0:
                self.model.save(self.save_path + "_" + str(self.env.total_steps))

            # plot avg reward per step and cumulative reward in tensorboard
            self.logger.record("reward/return", cumul_reward)
            self.logger.record("reward/mean_reward", (cumul_reward / (0.001 + self.env.total_steps)))
        return True


# REINFORCEMENT LEARNING TRAINING
def train(agent_type, policy, total_timesteps, model_name, log_dir, budget_reset="episodic"):
    os.makedirs(log_dir, exist_ok=True)
    # initialize the environment
    # if policy.endswith("LstmPolicy") or policy.endswith("LnLstmPolicy"):
    #     env = DummyVecEnv([lambda: Monitor(fiveG_net(policy), log_dir)])
    # else:

    # env = gym.wrappers.FlattenObservation(fiveG_net(policy))
    env_train = Monitor(MOfiveG_net(policy, budget_reset=budget_reset, non_MORL=True), log_dir)
    env_train.action_space.seed(123)

    # DEFINE AND TRAIN MODEL
    # Add some param noise for exploration
    # param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1) # (HYPER)
    # Create the callback: check every 10000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir, model_name=model_name, policy=policy, env=env_train) # (HYPER)
    print('start training the ', agent_type, ' agent')
    if agent_type == "DQN":
        model = DQN(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "A2C":
        model = A2C(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "PPO":
        model = PPO(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "DDPG":
        model = DDPG(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "SAC":
        model = SAC(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "TD3":
        model = TD3(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    else: #MaskablePPO
        model = MaskablePPO(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    
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

    with open(log_dir+'Log'+model_name+'.txt','a') as f:
        with contextlib.redirect_stdout(f):
            model.learn(total_timesteps, callback=callback)

    plot_results(log_dir, "OptSFC " + agent_type + " Learning Curve").savefig(log_dir + 'plot_' + model_name + '.pdf')
    # save the model
    model.save(log_dir + model_name + "last")

    if env_train.env.explain_log:
        df_log     = pd.DataFrame(env_train.env.explain_log)
        match_rate = df_log["match"].mean() * 100
        print(f"Explanation match rate: {match_rate:.1f}%")
        print(f"   Matched: {df_log['match'].sum()} / {len(df_log)}")

    # del env_train, model
    # gc.collect()
    actual_env = env_train.env
    del model
    gc.collect()

    return actual_env


# functions for plotting
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = moving_average(y, window=2)
    # change y from the cumulative reward to the reward per episode
    y = np.diff(y)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    # plt.show()
    return plt


def initialize_model_for_flwr(rl_algo, log_dir, budget_reset):
    # RL configs for flwr
    policy = "MlpPolicy"
    train_dir = log_dir
    eval_dir = "./tested_models/"
    # initialize the environment
    if rl_algo in ["PPO", "A2C", "MaskablePPO"]:
        env_train = Monitor(MOfiveG_net("MlpPolicy", budget_reset, non_MORL=True), train_dir)
        # env_train = DummyVecEnv([lambda: env_train])
        # env_train = VecCheckNan(env_train, raise_exception=True)
        eval_env = MOfiveG_net("MlpPolicy", budget_reset, non_MORL=True)
    else:
        env_train = MOfiveG_net("MlpPolicy", budget_reset)
        eval_env = MOfiveG_net("MlpPolicy", budget_reset)
    env_train.action_space.seed(123)
    eval_env.action_space.seed(123)

    # initialize the model
    if rl_algo == "Envelope":
        gamma = 0.99
        lr = 3e-4
        epsilon = 0.01
        batch_size = 256
        net_arch = [256, 256, 256, 256]
        model = Envelope(env_train, learning_rate=lr, gamma=gamma, initial_epsilon=epsilon, final_epsilon=epsilon, batch_size=batch_size, net_arch=net_arch, log = False)
    elif rl_algo == "EUPG":
        model = EUPG(env_train, scalarization=scalarization, weights=np.array(rewards_coeff), gamma=0.99, log=False, learning_rate=0.001)
    elif rl_algo == "A2C":
        model = A2C(policy, env_train)
    elif rl_algo == "PPO":
        model = PPO(policy, env_train)
    elif rl_algo == "MaskablePPO":
        model = MaskablePPO(policy, env_train)

    return model, env_train, eval_env



def train_in_flwr(model, train_env, total_timesteps, model_name, log_dir):
    # RL configs for flwr
    policy = "MlpPolicy"
    train_dir = "./trained_models/"
    model_name = "PPO_model"

    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=train_dir, model_name=model_name, policy=policy, env=train_env) # (HYPER)

    with open(log_dir+'Log'+model_name+'.txt','a') as f:
        with contextlib.redirect_stdout(f):
            model.learn(total_timesteps, callback=callback)

    plot_results(log_dir, "OptSFC " + policy + " Learning Curve").savefig(log_dir + 'plot_' + model_name + '.pdf')

    return model, train_env


from gymnasium.envs.registration import register

register(id="MOfiveG_net-v0", entry_point="short_episodic_mo_fiveg_mdp:MOfiveG_net")