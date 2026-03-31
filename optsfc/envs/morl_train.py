import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import json
import random
from types import SimpleNamespace
import matplotlib.pyplot as plt

#local files
from .mo_fiveg_mdp import MOfiveG_net
from .short_simulated_testbed import impact_ssla_factors, is_action_possible

# SB3
from stable_baselines3 import A2C, PPO
from sb3_contrib import MaskablePPO

# MORL-BASELINES
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.envelope.envelope import Envelope
# from morl_baselines.multi_policy.morld.morld import MORLD
# from morl_baselines.single_policy.esr.eupg import EUPG
from optsfc.envs.eupg_explain import EUPG
import torch as th

rewards_coeff = [0.4, 0.3, 0.3]
division_factor = 30

# def scalarization(reward: np.ndarray, weights: np.ndarray = None) -> float:
#     if reward.ndim == 1 and reward.size == 3:
#         return float(reward[0] * rewards_coeff[0] + reward[1] * rewards_coeff[1] + reward[2] * rewards_coeff[2])
#
#     else:  # Case for multiple rewards
#         # Convert reward to a PyTorch tensor if it's a NumPy array
#         reward_tensor = th.tensor(reward) if isinstance(reward, np.ndarray) else reward
#
#         # Ensure the operation produces a scalar
#         min_value = th.min(reward_tensor[:, 0], reward_tensor[:, 1] // 2)
#         return float(min_value.item())  # Use .item() to extract scalar from tensor

def scalarization(reward: np.ndarray, weights= None) -> float:
    if reward.ndim == 1 and reward.size == 3:
        return float(reward[0] * rewards_coeff[0] + reward[1] * rewards_coeff[1] + reward[2] * rewards_coeff[2])
    elif reward.ndim > 1 and reward.shape[1] == 3:
        res = float(sum(reward[:, 0] * rewards_coeff[0] + reward[:, 1] * rewards_coeff[1] + reward[:, 2] * rewards_coeff[2]))
        print("res", res)
        return res


def eupg_model_save(model, save_dir, filename, save_replay_buffer=True):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    saved_params = {}
    saved_params["net_state_dict"] = model.net.state_dict()
    saved_params["weights"] = model.weights
    saved_params["global_step"] = model.global_step
    saved_params["net_optimizer_state_dict"] = model.optimizer.state_dict()
    if save_replay_buffer:
        saved_params["buffer"] = model.buffer
    filename = model.experiment_name if filename is None else filename
    th.save(saved_params, save_dir + "/" + filename + ".tar")


def eupg_model_load(path, model, load_replay_buffer=True):
    """Load the model and the replay buffer if specified.

    Args:
        path: Path to the model.
        load_replay_buffer: Whether to load the replay buffer too.
    """
    params = th.load(path)
    model.net.load_state_dict(params["net_state_dict"])
    model.optimizer.load_state_dict(params["net_optimizer_state_dict"])
    if load_replay_buffer and "buffer" in params:
        model.replay_buffer = params["buffer"]


# A static baseline policy
def pick(lst):
    normalized_lst = [x / sum(lst) for x in lst]
    r = random.uniform(0, 1)
    cumulative_sum = 0
    for i, probability in enumerate(normalized_lst):
        cumulative_sum += probability
        if r <= cumulative_sum:
            return i


def baseline_model(env):
    action = 0 # do nothing
    no_action = not np.any(env.observation['mtd_action'])
    # # if there is an MTD action ongoing do nothing
    # if not no_action:
    #     return action
    # x% of times do nothing
    if random.uniform(0, 1) < 0.6:
        return action
    # decide the VNF according to the SLA impact value
    target_vnf = pick(impact_ssla_factors)
    # decide on the MTD action type according to SSLA
    target_action = pick([env.reinstantiations_per_month, env.migrations_per_month]) + 1
    if target_action == 1:
        action = target_vnf * 2
    else:
        action = target_vnf * 2 + 1
    if is_action_possible(env.environment, action)[0]:
        return action
    else:
        return 0 # do nothing


def make_division_strategy(env):
    # get budgets of MTD actions from the environment
    vnf_migrations = int(env.migrations_per_month // division_factor)
    vnf_reinstantiations = int(env.reinstantiations_per_month // division_factor)
    stateful_migrations = int(env.stateful_migrations_per_month // division_factor)

    # make a list of the actions to be taken
    one_cycle_actions_list = [1, 3, 5, 7, 9, 10, 11, 2, 4, 6, 8] * stateful_migrations
    vnf_migrations -= stateful_migrations
    vnf_reinstantiations -= stateful_migrations
    second_cycle_actions_list = [1, 3, 5, 7, 2, 4, 6, 8] * vnf_migrations
    vnf_reinstantiations -= vnf_migrations
    third_cycle_actions_list = [ 2, 4, 6, 8] * vnf_reinstantiations
    # concatenate the lists
    actions_list = one_cycle_actions_list + second_cycle_actions_list + third_cycle_actions_list
    return actions_list


def division_strategy_action(env, actions_list):
    # measure break period between mtd phases
    month_timesteps = env.constraints_reset
    divided_month = month_timesteps // division_factor
    # get action_index as the remainder of the division of the step_counter by one_fifth_month
    action_index = int(env.step_counter % divided_month)
    # get the action from the list of actions if the action_index is less than the length of the list
    if action_index < len(actions_list):
        return actions_list[action_index]
    else:
        return 0


def plot_factorize_data(data, factor):
    return [np.mean(data[i:i + factor]) for i in range(0, len(data), factor)]


def plot_factorize_indices(indices, factor):
    result = [idx // factor for idx in indices]
    # remove duplicates. Order is not important
    return set((result))


def plot_test_results(filename, plot_rewards, plot_return, plot_steps_with_null_actions, plot_episode_ends, avg_obs_reward):
    """"
        Plot with steps at x axis (the steps are the same as the length of the rewards list).
        The y axis is the reward/return value.
        the plot_steps_with_null_actions is a list of steps where null actions were taken.
        The plot contains the rewards and the return values each in a line of different color.
        The null actions are marked with red dots on the x axis, while the y value is taken from the rewards list.
        plot episode ends are marked with vertical lines.
        The plot is saved in a pdf file.
    """
    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    plot_rewards = plot_factorize_data(plot_rewards, 500)
    plot_return = plot_factorize_data(plot_return, 500)
    plot_episode_ends = plot_factorize_indices(plot_episode_ends, 500)
    plot_steps_with_null_actions = plot_factorize_indices(plot_steps_with_null_actions, 500)

    # Plot rewards
    plt.plot(range(len(plot_rewards)), plot_rewards, label="Rewards", color='blue', linewidth=1.5)

    # Plot return values
    plt.plot(range(len(plot_return)), plot_return, label="Returns", color='green', linestyle='--', linewidth=1.5)

    # Mark episode ends with vertical lines
    for episode_end in plot_episode_ends:
        plt.axvline(x=episode_end, label="Episode", color='black', linestyle='--', linewidth=0.5)

    # Mark null actions with red dots
    for null_action in plot_steps_with_null_actions:
        plt.scatter(null_action, plot_rewards[null_action], color='red', s=10)

    # Add labels, title, and legend
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.title("Test Results: Rewards and Returns (avg. rew:" + str(avg_obs_reward) + " )", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Save the plot to a PDF file
    plt.tight_layout()
    plt.savefig(str(filename) + ".pdf")
    plt.close()


def eval_mo_reward_conditioned(
    agent,
    model_name,
    env,
    test_timesteps,
    filename,
    scalarization,
    w: np.ndarray = None,
    render: bool = False,
    verbose: bool =False
) -> [float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment. This makes the assumption that the agent is conditioned on the accrued reward i.e. for ESR agent.
    Args:
        agent: Agent
        env: mo_gymnasium environment
        scalarization: scalarization function, taking weights and reward as parameters
        w: weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized total reward, scalarized return, vectorized total reward, vectorized return
    """
    obs, _ = env.reset()
    mono_objective = False
    return_value = 0
    vec_return, disc_vec_return = np.zeros(env.reward_space.shape[0]), np.zeros(env.reward_space.shape[0])
    gamma = 1.0
    n_steps = test_timesteps
    division_strategy_list = make_division_strategy(env)
    action = None
    previous_action = None
    # counters for the test
    null_actions = 0
    test_tot = 0
    total_jumped_steps = 0
    list_cumul_rew = []
    list_null_actions = []
    log_actions = [ 0 for _ in range (env.n_actions)]
    # counters for the episode
    episode_log_actions = [ 0 for _ in range (env.n_actions)]
    episode_steps = 0
    episode_rewards = 0
    episode_null_actions = 0
    episodes = 0
    # variables for plotting
    plot_rewards = []
    plot_return = []
    plot_actions = []
    plot_episode_ends = []

    for step in range(n_steps):
        episode_steps += 1
        if verbose: # print some info
            if step % 50000 == 0:
                print("At step ", step, " total reward is: ", test_tot)
                print("Total reward difference: ", vec_return - test_tot)
                print("Max penalty values: resource_utilization: ", env.max_resource_penalty, "network_penalty: ",
                      env.max_network_penalty, "security_penalty: ", env.max_security_penalty)
        if agent.experiment_name == "random":
            # the action is random and also has to be valid
            while True:
                action = env.action_space.sample()
                if is_action_possible(env.environment, action)[0]:
                    break
        elif agent.experiment_name == "static":
            action = baseline_model(env)
        elif agent.experiment_name == "nothing":
            action = 0
        elif agent.experiment_name == "same":
            action = 1
        elif agent.experiment_name == "division":
            action = division_strategy_action(env, division_strategy_list)
        else:
            same = 0
            counter = 0
            if action != None:
                previous_action = action
            while counter < 120:
                counter += 1
                if agent.experiment_name == "EUPG":
                    action = agent.eval(obs, vec_return)
                elif agent.experiment_name == "Envelope":
                    action = agent.eval(obs, rewards_coeff)
                else:
                    mono_objective = True
                    action = agent.predict(observation=obs, deterministic=True)[0]

                if previous_action != None and previous_action == action:
                    same += 1
                else:
                    same = 0
                if same > 10 and not is_action_possible(env.environment, action)[0]:
                    print("same invalid action selected 10 consecutive times")
                    action = 0
                    break
                if is_action_possible(env.environment, action)[0]:
                    break

        log_actions[action] += 1
        episode_log_actions[action] += 1
        plot_actions.append(action)
        if action == 0:
            null_actions += 1
            episode_null_actions += 1
            list_null_actions.append(step)

        obs, r, done, truncated, info = env.step(action)

        if mono_objective:
            return_value += info["rew"]
        else:
            # sum vector r to vec_return by index
            vec_return[0] += r[0]
            vec_return[1] += r[1]
            vec_return[2] += r[2]
            if agent.experiment_name not in ["random", "static", "nothing", "same", "division"]:
                disc_vec_return[0] += gamma * r[0]
                disc_vec_return[1] += gamma * r[1]
                disc_vec_return[2] += gamma * r[2]
                gamma *= agent.gamma
        episode_rewards += info["rew"]

        if done:
            episodes += 1
            if verbose:
                print(episodes, "The episode ended. Environment resetted.")
            episode_jumped_steps = int(info["jumped_steps"])
            for _ in range(episode_jumped_steps):
                plot_rewards.append(info["rew"] / episode_jumped_steps)
                test_tot += info["rew"] / episode_jumped_steps
                plot_return.append(test_tot)
            total_jumped_steps += episode_jumped_steps
            plot_episode_ends.append(step + total_jumped_steps)
            # print(agent.experiment_name, "episodes=", episodes, "tot steps=", episode_steps,
            #       "decided_null_actions_rate=",  0 if episode_null_actions == 0 else episode_null_actions / episode_steps,
            #       "null_actions_rate=", 0 if episode_null_actions == 0 else (episode_null_actions + episode_jumped_steps) / (episode_steps + episode_jumped_steps),
            #       "steps jumped=", episode_jumped_steps,
            #       " avg_obs_reward", episode_rewards / (episode_steps + episode_jumped_steps),
            #       " avg_without_last_reward", (episode_rewards - episode_jumped_steps) / (episode_steps + episode_jumped_steps),
            #       "log of actions", log_actions)
            # reset counters
            obs, info = env.reset()
            episode_steps = 0
            episode_rewards = 0
            episode_null_actions = 0
            episode_log_actions = [ 0 for _ in range (env.n_actions)]
        else:
            plot_rewards.append(info["rew"])
            test_tot += info["rew"]
            plot_return.append(test_tot)

        list_cumul_rew.append(test_tot)
    if agent.experiment_name == "EUPG" or agent.experiment_name == "Envelope" or agent.experiment_name == "PQL":
        if w is None:
            scalarized_return = scalarization(vec_return)
            scalarized_discounted_return = scalarization(disc_vec_return)
        else:
            scalarized_return = scalarization(w, vec_return)
            scalarized_discounted_return = scalarization(w, disc_vec_return)
    else:
        scalarized_return = return_value
        scalarized_discounted_return = return_value

    #save list_cumul_rew and list_null_actions in JSON file
    avg_obs_reward = test_tot / (step + total_jumped_steps)
    output = {"list_cumul_rew": list_cumul_rew, "list_returns": plot_return, "list_rewards": plot_rewards, "episodes": episodes, "null_actions_rate": 0 if null_actions == 0 else (step + total_jumped_steps) / (null_actions + total_jumped_steps), 'avg_obs_reward(bench)': avg_obs_reward, 'avg_discounted_reward': scalarized_return / (step + total_jumped_steps), 'log_actions': log_actions}
    with open(filename, 'w') as f:
        json.dump(output, f)
    print("find logs in " + filename)

    if verbose:
        print("total steps= ", step + total_jumped_steps, "episodes= ", episodes,
              "decided_null_actions_rate=", 0 if null_actions == 0 else null_actions / step,
              "null_actions_rate= ", 0 if null_actions == 0 else (null_actions + total_jumped_steps) / (step + total_jumped_steps), 'avg_obs_reward(bench)= ', avg_obs_reward, 'avg_discounted_reward= ', scalarized_return / (step + total_jumped_steps), 'log_actions= ', log_actions)

    # plot the rewards, return, and actions
    plot_test_results(model_name + agent.experiment_name, plot_rewards, plot_return, list_null_actions, plot_episode_ends, avg_obs_reward)

    return [
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    ]


def train_pql():
    env = MOfiveG_net("MlpPolicy")
    ref_point = np.array([0, -25])

    agent = PQL(
        env,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.997,
        final_epsilon=0.2,
        seed=1,
        log=False,
    )

    # Training
    pf = agent.train(num_episodes=1000, log_every=100, action_eval="pareto_cardinality")#"hypervolume")
    assert len(pf) > 0

    # Policy following
    target = np.array(pf.pop())
    tracked = agent.track_policy(target)
    assert np.all(tracked == target)


def train_Envelope(total_timesteps, model_name, budget_reset="episodic", gamma=0.99, lr=3e-4, epsilon=0.01, batch_size=256, net_arch=[256, 256, 256, 256]):
    env = MOfiveG_net("MlpPolicy", budget_reset)

    save_replay_buffer = True
    filename = model_name
    save_dir = "models"

    # Train the agent
    agent = Envelope(env, learning_rate=lr, gamma=gamma, initial_epsilon=epsilon, final_epsilon=epsilon, batch_size=batch_size, net_arch=net_arch, log = False)
    
    env.model_for_explain = agent

    agent.train(total_timesteps= total_timesteps, eval_freq=1000)
    agent.save(save_dir=save_dir, filename=filename, save_replay_buffer= save_replay_buffer)

    if env.explain_log:
        import pandas as pd
        df_log = pd.DataFrame(env.explain_log)
        match_rate = df_log["match"].mean() * 100
        print(f"Match rate: {match_rate:.1f}%")
        print(f"   Matched: {df_log['match'].sum()} / {len(df_log)}")
    return env


def split_train_Envelope(total_timesteps, timesteps_split, model_name, budget_reset="episodic", gamma=0.99, lr=3e-4, epsilon=0.01, batch_size=256, net_arch=[256, 256, 256, 256]):
    env = MOfiveG_net("MlpPolicy", budget_reset)

    save_replay_buffer = True
    filename = model_name
    save_dir = "models"

    # Train the agent
    agent = Envelope(env, learning_rate=lr, gamma=gamma, initial_epsilon=epsilon, final_epsilon=epsilon, batch_size=batch_size, net_arch=net_arch, log = True)
    # train in a loop and save the model after each split
    for i in range(0, total_timesteps, timesteps_split):
        agent.train(total_timesteps=timesteps_split, eval_freq=1000)
        filename = model_name + str(i)
        agent.save(save_dir=save_dir, filename=filename, save_replay_buffer=save_replay_buffer)


def train_eupg(total_timesteps, model_name, budget_reset="episodic"):
    env = MOfiveG_net("MlpPolicy", budget_reset)
    eval_env = MOfiveG_net("MlpPolicy", budget_reset)

    save_dir = "models"
    filename = model_name
    # convert rewards_coeff list into np.array
    weights = np.array(rewards_coeff)

    agent = EUPG(env, scalarization=scalarization, weights=weights, gamma=0.99, log=False, learning_rate=0.001)

    env.model_for_explain = agent

    agent.train(total_timesteps=total_timesteps, eval_env=eval_env)
    eupg_model_save(agent, save_dir, filename)

    if env.explain_log:
        import pandas as pd
        df_log     = pd.DataFrame(env.explain_log)
        match_rate = df_log["match"].mean() * 100
        print(f"Explanation match rate: {match_rate:.1f}%")
        print(f"   Matched: {df_log['match'].sum()} / {len(df_log)}")
    return env


def split_train_eupg(total_timesteps, timesteps_split, model_name, budget_reset="episodic"):
    env = MOfiveG_net("MlpPolicy", budget_reset)
    eval_env = MOfiveG_net("MlpPolicy", budget_reset)

    save_dir = "models"
    # convert rewards_coeff list into np.array
    weights = np.array(rewards_coeff)

    agent = EUPG(env, scalarization=scalarization, weights=weights, gamma=0.99, log=True, learning_rate=0.001)
     # train in a loop and save the model after each split
    for i in range(0, total_timesteps, timesteps_split):
        agent.train(total_timesteps=timesteps_split, eval_env=eval_env)
        filename = model_name + str(i)
        eupg_model_save(agent, save_dir, filename)



def eval_agent_split(test_timesteps, total_timesteps, timestep_split, agent_type, model_name, budget_reset="episodic", verbose=False):
    if budget_reset != "episodic":
        env = MOfiveG_net("MlpPolicy", budget_reset=budget_reset)
    else:
        env = MOfiveG_net("MlpPolicy")

    # init agent and load the model based on the agent type
    save_dir = "./models"
    save_replay_buffer = True

    for i in range(0, total_timesteps, timestep_split):
        if agent_type == "Envelope":
            agent = Envelope(env, log = False)
            if i > 0:
                agent.load(path=save_dir + '/' + model_name + str(i) + '.tar', load_replay_buffer=save_replay_buffer)
        elif agent_type == "EUPG":
            agent = EUPG(env, scalarization=scalarization, log= False)
            if i > 0:
                eupg_model_load(path=save_dir + '/' + model_name + str(i) + '.tar', model=agent)
        elif agent_type in ["A2C", "PPO", "MaskablePPO"]:
            if i == 0:
                model_path = save_dir + '/' + model_name + '/best_model.zip'
            else:
                model_path = save_dir + '/' + model_name + '/best_model_' + str(i) + '.zip'
            # after defining the model name based on the split load the model based on the agent type
            if agent_type == "A2C":
                agent = A2C.load(model_path)
            elif agent_type == "PPO":
                agent = PPO.load(model_path)
            elif agent_type == "MaskablePPO":
                agent = MaskablePPO.load(model_path)
        else:
            agent = SimpleNamespace()
        agent.experiment_name = agent_type
        if verbose:
            print("model ", model_name," loaded. \n Model Evaluation:")

        json_and_plot_filename = model_name + "_split" +str(i)+"_test" + str(test_timesteps)
        scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = eval_mo_reward_conditioned(
            agent, model_name=json_and_plot_filename, env=env, scalarization=scalarization, test_timesteps=test_timesteps,
            filename="./cumul_rewards/"+ json_and_plot_filename + ".json", verbose=verbose)


def eval_agent(total_timesteps, agent_type, model_name, budget_reset="episodic", verbose=False):
    if budget_reset != "episodic":
        env = MOfiveG_net("MlpPolicy", budget_reset=budget_reset)
    else:
        env = MOfiveG_net("MlpPolicy")

    # init agent and load the model based on the agent type
    save_dir = "./models"
    save_replay_buffer = True
    if agent_type == "Envelope":
        agent = Envelope(env, log = False)
        agent.load(path=save_dir + '/' + model_name + '.tar', load_replay_buffer=save_replay_buffer)
    elif agent_type == "PQL":
        agent = PQL(env)
    elif agent_type == "EUPG":
        agent = EUPG(env, scalarization=scalarization, log= False)
        eupg_model_load(path=save_dir + '/' + model_name + '.tar', model=agent)
    elif agent_type == "A2C":
        agent = A2C.load(save_dir + '/' + model_name + '/best_model.zip')
    elif agent_type == "PPO":
        agent = PPO.load(save_dir + '/' + model_name + '/best_model.zip')
    elif agent_type == "MaskablePPO":
        agent = MaskablePPO.load(save_dir + '/' + model_name + '/best_model.zip')
    else:
        agent = SimpleNamespace()
    agent.experiment_name = agent_type
    if verbose:
        print("model ", model_name," loaded. \n Model Evaluation:")

    scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = eval_mo_reward_conditioned(
        agent, model_name=model_name, env=env, scalarization=scalarization, test_timesteps=total_timesteps, filename="./cumul_rewards/"+ model_name + agent.experiment_name +"_"+str(total_timesteps)+".json", verbose=verbose)


def transfer_learning(total_timesteps, agent_type, policy, model_name, previous_model_name, log_dir, budget_reset="episodic", verbose=False):
    if budget_reset != "episodic":
        env = MOfiveG_net("MlpPolicy", budget_reset=budget_reset)
    else:
        env = MOfiveG_net("MlpPolicy")

    # init agent and load the model based on the agent type
    save_dir = "./models"
    save_replay_buffer = True
    if agent_type == "Envelope":
        agent = Envelope(env, log = False)
        agent.load(path=save_dir + '/' + previous_model_name + '.tar', load_replay_buffer=save_replay_buffer)
    elif agent_type == "PQL":
        agent = PQL(env)
    elif agent_type == "EUPG":
        agent = EUPG(env, scalarization=scalarization, log= False)
        eupg_model_load(path=save_dir + '/' + previous_model_name + '.tar', model=agent)
    elif agent_type == "A2C":
        agent = A2C.load(save_dir + '/' + previous_model_name + '/best_model.zip')
    elif agent_type == "PPO":
        agent = PPO.load(save_dir + '/' + previous_model_name + '/best_model.zip')
    elif agent_type == "MaskablePPO":
        agent = MaskablePPO.load(save_dir + '/' + previous_model_name + '/best_model.zip')
    else:
        agent = SimpleNamespace()
    agent.experiment_name = agent_type
    if verbose:
        print("model ", previous_model_name," loaded. \n Continue its training to reach " + model_name)

    if agent_type in ["A2C", "PPO", "MaskablePPO"]:
        agent.learn(total_timesteps=total_timesteps, callback=None, seed=None, reset_num_timesteps=False, tensorboard_log="./tmp/"+model_name+"/" )
