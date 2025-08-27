import random
import copy
import time
import math
import numpy as np
from .short_space_dict import locations, vnfs_size, cnfs_size


#HYPERPARAMETERS
hard_action_proactive_reward = 1.5
one_step_duration = 40
recon_asp_increase_factor = 0.05
zero_day_asp = 0.01

latency_sla_penalty_factor = 1.1
impact_ssla_factors = [1, 1.2, 1.5, 2] # map impact_ssla category (the index list) to a factor (the value)

#STATISTICAL DATA FOR THE TRAFFIC SIMULATION BASED ON TESTBED MEASUREMENTS
edge_latency_1ue = {'min': 0.0026, 'max': 0.0039, 'avg':0.0031, 'std':0.00024}
core_latency_1ue = {'min': 0.003, 'max': 0.0044, 'avg': 0.0036, 'std': 0.000206}
edge_throughput_1ue = {'min':3212446.807, 'max':4618873.5705, 'avg':3955135.7566, 'std':288652.887}
core_throughput_1ue = {'min':2794515.509, 'max': 4093509.573, 'avg': 3445366.400, 'std': 276464.785}

# this is used only to compute the increase by ue connection
edge_latency_10ue = {'min': 0.01334, 'max': 0.0189, 'avg': 0.0153, 'std':0.001}
core_latency_10ue = {'min': 0.0132, 'max': 0.01947, 'avg': 0.0149, 'std': 0.0009}
edge_throughput_10ue = {'min':787710.007, 'max':1125071.197, 'avg':949175.4814, 'std':54341.243}
core_throughput_10ue = {'min':827971.609, 'max':1293276.664, 'avg':965303.2339, 'std':58015.922}

# compute statistical increase/decrease between 10ue values and 1ue values
additional_ue_latency_increase_edge = {}
additional_ue_latency_increase_core = {}
additional_ue_throughput_decrease_edge = {}
additional_ue_throughput_decrease_core = {}
for key in edge_latency_10ue.keys():
    if key == 'max':
        additional_ue_latency_increase_edge[key] = (edge_latency_10ue[key] - edge_latency_1ue['min']) / 10
        additional_ue_latency_increase_core[key] = (core_latency_10ue[key] - core_latency_1ue['min']) / 10
        additional_ue_throughput_decrease_edge[key] = (edge_throughput_10ue[key] - edge_throughput_1ue['min']) / 10
        additional_ue_throughput_decrease_core[key] = (core_throughput_10ue[key] - core_throughput_1ue['min']) / 10
    elif key == 'min':
        additional_ue_latency_increase_edge[key] = (edge_latency_1ue['max'] - edge_latency_10ue[key]) / 10
        additional_ue_latency_increase_core[key] = (core_latency_1ue['max'] - core_latency_10ue[key]) / 10
        additional_ue_throughput_decrease_edge[key] = (edge_throughput_1ue['max'] - edge_throughput_10ue[key]) / 10
        additional_ue_throughput_decrease_core[key] = (core_throughput_1ue['max'] - core_throughput_10ue[key]) / 10
    else:
        additional_ue_latency_increase_edge[key] = (edge_latency_10ue[key] - edge_latency_1ue[key]) / 10
        additional_ue_latency_increase_core[key] = (core_latency_10ue[key] - core_latency_1ue[key]) / 10
        additional_ue_throughput_decrease_edge[key] = (edge_throughput_10ue[key] - edge_throughput_1ue[key]) / 10
        additional_ue_throughput_decrease_core[key] = (core_throughput_10ue[key] - core_throughput_1ue[key]) / 10

# STATISTICAL DATA FOR MTD ACTIONS ON STATELESS VNFs
edge_mtd_migration_time = {'min': 1.52, 'max': 1.5943, 'avg': 1.558, 'std':0.0237}
core_mtd_migration_time = {'min': 1.0546, 'max': 1.2687, 'avg': 1.1545, 'std':0.078}
edge_mtd_restart_time = {'min': 1.884, 'max': 2.562, 'avg':2.2996, 'std':0.212}
core_mtd_restart_time = {'min': 0.999, 'max': 1.3464, 'avg': 1.0956, 'std':0.1126}

restart_qos_overhead_rate = {"latency": 2, "packet_loss_rate":0.017}
restart_ploss_rate_duration = 1 # in seconds
restart_latency_duration = 1

migrate_qos_overhead_rate = {"latency": 5, "packet_loss_rate":0.08}
migrate_ploss_rate_duration = 1 # in seconds
migrate_latency_duration = 1.5

# STATISTICAL DATA FOR MTD ACTIONS ON STATEFUL CNFs
container_mtd_migration_time = {'min': 1.129, 'max': 32.07, 'avg': 8.99, 'std':7.59}
core_container_mtd_transfer_time = {'min': 0.4, 'max': 0.6, 'avg': 0.5, 'std':0.05}


def get_statistical_value(statistical_data):
    mean = statistical_data['avg']
    std = abs(statistical_data['std'])
    result = random.gauss(mean, std)
    return min(statistical_data["max"], max(statistical_data["min"], result))


def min_to_sec(min):
    return min * 60


# calculate restart with migration_time - transfer_time formula
# def get_container_restart_time():
#     # get a migration time from the statistical data
#     migration_time = get_statistical_value(container_mtd_migration_time)
#     # get a transfer time from the statistical data
#     transfer_time = get_statistical_value(core_container_mtd_transfer_time)
#     return migration_time - transfer_time


def generate_mtd_action_time(new_obs, vnf_index, is_cnf=False):
    # if it's a cnf we use the container migration time
    if is_cnf:
        new_obs['mtd_action'][vnf_index][1] = min_to_sec(get_statistical_value(container_mtd_migration_time))
        return

    # get the location of the vnf
    if new_obs['location'][vnf_index][0] == locations['edge']:
        if new_obs['mtd_action'][vnf_index][0] == 1: # restart in edge
            new_obs['mtd_action'][vnf_index][1] = min_to_sec(get_statistical_value(edge_mtd_restart_time))
        else: #migrate
            new_obs['mtd_action'][vnf_index][1] = min_to_sec(get_statistical_value(edge_mtd_migration_time))
    else:
        if new_obs['mtd_action'][vnf_index][0] == 1:  # restart in core
            new_obs['mtd_action'][vnf_index][1] = min_to_sec(get_statistical_value(core_mtd_restart_time))
        else:  # migrate
            new_obs['mtd_action'][vnf_index][1] = min_to_sec(get_statistical_value(core_mtd_migration_time))


def simulate_restart(new_obs, vnf_index):
    # add the network overhead to the vnf
    latency_overhead = restart_qos_overhead_rate['latency'] * (restart_latency_duration / one_step_duration)
    ploss_rate_overhead = restart_qos_overhead_rate['packet_loss_rate'] * (restart_ploss_rate_duration / one_step_duration)
    new_obs["network_metrics"][vnf_index][0] += latency_overhead
    new_obs["network_metrics"][vnf_index][2] += ploss_rate_overhead
    new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][0] += new_obs['resource_consumption'][vnf_index][0]
    new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][1] += new_obs['resource_consumption'][vnf_index][1]
    new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][2] += new_obs['resource_consumption'][vnf_index][2]


def simulate_migrate(new_obs, vnf_index):
    # add the network overhead to the vnf
    latency_overhead = migrate_qos_overhead_rate['latency'] * (migrate_latency_duration / one_step_duration)
    ploss_rate_overhead = migrate_qos_overhead_rate['packet_loss_rate'] * (migrate_ploss_rate_duration / one_step_duration)
    new_obs["network_metrics"][vnf_index][0] += latency_overhead
    new_obs["network_metrics"][vnf_index][2] += ploss_rate_overhead

    # free allocated resources in previous VIM
    # print('log before: Resources of VIM', new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]],
    #           ' reallocated after finishing for "migrate" vnf', vnf_index)
    new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][0] += new_obs['resource_consumption'][vnf_index][0]
    new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][1] += new_obs['resource_consumption'][vnf_index][1]
    new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][2] += new_obs['resource_consumption'][vnf_index][2]
    # print('log after: Resources of VIM', new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]],
    #           ' reallocated after finishing for "migrate" vnf', vnf_index)

    # update vnf to new location
    if new_obs["location"][vnf_index][0] == locations['edge']:
        new_obs["location"][vnf_index][0] = locations['core']
        new_obs["vim_host"][vnf_index][0] = locations['core']
    else:
        new_obs["location"][vnf_index][0] = locations['edge']
        new_obs["vim_host"][vnf_index][0] = locations['edge']


def simulate_stateful_migrate(new_obs, cnf_index):
    # free allocated resources in previous VIM
    new_obs['vim_resources'][new_obs['vim_host'][cnf_index][0]][0] += new_obs['resource_consumption'][cnf_index][0]
    new_obs['vim_resources'][new_obs['vim_host'][cnf_index][0]][1] += new_obs['resource_consumption'][cnf_index][1]
    new_obs['vim_resources'][new_obs['vim_host'][cnf_index][0]][2] += new_obs['resource_consumption'][cnf_index][2]

    if new_obs["vim_host"][cnf_index][0] == locations['edge']:
        new_location = locations['core']
    else:
        new_location = locations['edge']

    # allocate resources in the new VIM
    new_obs['vim_resources'][new_location][0] -= new_obs['resource_consumption'][cnf_index][0]
    new_obs['vim_resources'][new_location][1] -= new_obs['resource_consumption'][cnf_index][1]
    new_obs['vim_resources'][new_location][2] -= new_obs['resource_consumption'][cnf_index][2]

    # update cnf to new location
    if new_obs["location"][cnf_index][0] == locations['edge']:
        new_obs["location"][cnf_index][0] = locations['core']
        new_obs["vim_host"][cnf_index][0] = locations['core']
    else:
        new_obs["location"][cnf_index][0] = locations['edge']
        new_obs["vim_host"][cnf_index][0] = locations['edge']


def set_double_resource_allocation(new_obs, vnf_index, action_type):
    if action_type == "restart":
        # add the resource overhead to the vnf's VIM host
        new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][0] -= new_obs['resource_consumption'][vnf_index][0]
        new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][1] -= new_obs['resource_consumption'][vnf_index][1]
        new_obs['vim_resources'][new_obs['vim_host'][vnf_index][0]][2] -= new_obs['resource_consumption'][vnf_index][2]
    else:
        # get new VIM location
        if new_obs["vim_host"][vnf_index][0] == locations['edge']:
            new_location = locations['core']
        else:
            new_location = locations['edge']
        # add the resource overhead to the vnf
        new_obs['vim_resources'][new_location][0] -= new_obs['resource_consumption'][vnf_index][0]
        new_obs['vim_resources'][new_location][1] -= new_obs['resource_consumption'][vnf_index][1]
        new_obs['vim_resources'][new_location][2] -= new_obs['resource_consumption'][vnf_index][2]


def get_simulated_value(statistical_data, statistical_increase, ues_number):
    # return random number using mean and standard deviation and bounded by min and max
    result = get_statistical_value(statistical_data)
    # get a random number for the increase from the statistical increase
    increase = get_statistical_value(statistical_increase) * ues_number
    return max(result + increase, 0)


def get_random_network_metrics(location, ues_number):
    # Get random network metric for a given location and number of UEs

    if location == locations['edge']:
        # get a random value based on the average, std, min and max in edge_latency_1ue
        latency = get_simulated_value(edge_latency_1ue, additional_ue_latency_increase_edge, ues_number)
        throughput = get_simulated_value(edge_throughput_1ue, additional_ue_throughput_decrease_edge, ues_number)
    else:
        latency = get_simulated_value(core_latency_1ue, additional_ue_latency_increase_core, ues_number)
        throughput = get_simulated_value(core_throughput_1ue, additional_ue_throughput_decrease_core, ues_number)
    p_loss = 0
    nb_packets = int(1 / latency)
    nb_packets_in = random.randrange(0, nb_packets + 1)
    nb_packets_out = nb_packets - nb_packets_in
    return latency, throughput, p_loss, nb_packets_in, nb_packets_out


def add_or_no():
    #       get random value  to 0 with 85% probability, to 1 with 7.5% probability, and -1 the rest
    r = random.choices([0,1,-1], weights=[85,7.5,7.5])
    return r[0]


def get_new_simulated_observation(new_obs):
    """"
        (second stage) from the first network setup
           we have 85% of doing nothing
           7.5% of adding a vnf
           7.5% of removing a vnf
    """

    # for each vnf simulate its traffic (throughput, latency) based on location
    for vnf_index in range(0, new_obs['nb_resources'][0]):
        # simulate the increase or decrease of connected ues
        added_ues = add_or_no()
        if new_obs['nb_UEs_cnx'][vnf_index][0] > 0:
            new_obs['nb_UEs_cnx'][vnf_index][0] = int(new_obs['nb_UEs_cnx'][vnf_index][0] + int(added_ues))

        # get the network metrics from get_random_network_metric
        new_obs['network_metrics'][vnf_index] = get_random_network_metrics(new_obs['location'][vnf_index][0], new_obs['nb_UEs_cnx'][vnf_index][0])

        # check that the vnf is not in a stateful container
        if new_obs['id'][vnf_index][0] >= 100:
            is_cnf = True
        else:
            is_cnf = False
        # if new_obs['location'][vnf_index][0] == locations['edge']:
        # check if an mtd action is in the queue to simulate its effect at the end and remove it
        if new_obs['mtd_action'][vnf_index][0] != 0:
            if new_obs['mtd_action'][vnf_index][1] == 0:
                #the action is new, randomly select the mtd action duration
                generate_mtd_action_time(new_obs, vnf_index, is_cnf)
                if new_obs['mtd_action'][vnf_index][0] == 1:
                    action_type = "restart"
                else:
                    action_type = "migrate"
                if not is_cnf: # vnfs are stateless and have double resource allocation
                    set_double_resource_allocation(new_obs, vnf_index, action_type)
            else:
                if is_cnf: # a cnf network overhead is calculated based on migration duration (new_obs['mtd_action'][vnf_index][1])
                    time_factor = min(new_obs['mtd_action'][vnf_index][1], one_step_duration) / one_step_duration
                    latency_overhead = migrate_qos_overhead_rate['latency'] * time_factor
                    ploss_rate_overhead = migrate_qos_overhead_rate['packet_loss_rate'] * time_factor
                    new_obs["network_metrics"][vnf_index][0] += latency_overhead
                    new_obs["network_metrics"][vnf_index][2] += ploss_rate_overhead

                new_obs['mtd_action'][vnf_index][1] -= one_step_duration
                if new_obs['mtd_action'][vnf_index][1] <= one_step_duration: # mtd action is finished
                    # simulate mtd effect
                    if new_obs['mtd_action'][vnf_index][0] == 1: # action is a restart
                        simulate_restart(new_obs, vnf_index)
                    else: # action is a migrate
                        if is_cnf:
                            simulate_stateful_migrate(new_obs, vnf_index)
                        else:
                            simulate_migrate(new_obs, vnf_index)
                    # remove mtd action once finished
                    new_obs['mtd_action'][vnf_index][0] = 0
                    new_obs['mtd_action'][vnf_index][1] = 0


def is_action_possible(observation, action_num):
    # print("action selected is ", str(action_num))
    # on every vnf 2 actions can be applied
    if action_num > (vnfs_size * 2 + cnfs_size + 1):
        return False, "there is no VNF running in the targetted row"

    if action_num == 0: # do nothing
        return True, None

    max_vnf_actions = vnfs_size * 2
    # for the vnf targetted check that there is an MTD already
    if action_num <= max_vnf_actions: # it's an action on a VNF
        vnf_index = int((action_num - 1) /2)
    else: # it's an action on a CNF
        vnf_index = None
        cnf_index = ((action_num - max_vnf_actions) + vnfs_size) - 1 # removed vnf actions and -1 for the index starting from 0

    if vnf_index is not None:
        if observation['mtd_action'][vnf_index][0] != 0:
            return False, "there is already an MTD in progress in VNF " + str(vnf_index)

        # check that the limit of MTDs possible is not reached
        if action_num % 2 == 0:
            # action is a restart
            if observation['mtd_constraint'][vnf_index][1] == 0:
                return False, "the limit of MTD restarts for VNF "+str(vnf_index)+" is reached"
        else:
            # action is a migrate
            if observation['mtd_constraint'][vnf_index][0] == 0:
                return False, "the limit of MTD migrates for VNF "+str(vnf_index)+" is reached"

        # check that the amount of cpu, ram and disk needed are available
        if observation['resource_consumption'][vnf_index][0] < observation['vim_resources'][observation['vim_host'][vnf_index][0]][0] and observation['resource_consumption'][vnf_index][1] < observation['vim_resources'][observation['vim_host'][vnf_index][0]][1] and observation['resource_consumption'][vnf_index][2] < observation['vim_resources'][observation['vim_host'][vnf_index][0]][2]:
            return True, None
        else:
            return False, "there is not enough resource to run the action"
    else:
        # print("action is on a CNF, action: ", action_num, "cnf index: ", cnf_index)
        if observation['mtd_action'][cnf_index][0] != 0:
            return False, "there is already an MTD in CNF" + str(cnf_index)

        # check that the limit of MTDs possible is not reached
        if observation['mtd_constraint'][cnf_index][0] == 0:
            return False, "the limit of MTD migrates for CNF " + str(cnf_index) + " is reached"
        # check that the amount of cpu, ram and disk needed are available
        if observation['resource_consumption'][cnf_index][0] < observation['vim_resources'][observation['vim_host'][cnf_index][0]][0] and observation['resource_consumption'][cnf_index][1] < observation['vim_resources'][observation['vim_host'][cnf_index][0]][1] and observation['resource_consumption'][cnf_index][2] < observation['vim_resources'][observation['vim_host'][cnf_index][0]][2]:
            return True, None
        else:
            return False, "there is not enough resource to run the action"


def perform_action(env, observation, action_num,  reward):
    if action_num == 0:
        return # action do nothing chosen

    max_vnf_actions = vnfs_size * 2
    # for the vnf targetted check that there is an MTD already
    if action_num <= max_vnf_actions:  # it's an action on a VNF
        vnf_index = int((action_num - 1) / 2)
    else:  # it's an action on a CNF
        vnf_index = None
        cnf_index = ((action_num - max_vnf_actions) + vnfs_size) - 1

    if vnf_index is not None:
        if action_num % 2 == 0:
            # action is a restart
            observation['mtd_action'][vnf_index][0] = 1
            observation['mtd_constraint'][vnf_index][1] -= 1 # reduce restart limit counter
        else:
            # action is a migrate
            observation['mtd_action'][vnf_index][0] = 2
            observation['mtd_constraint'][vnf_index][0] -= 1 # reduce migrate limit counter
        # reset asp of reconnaissance
        env.dynamic_asp[vnf_index]['recon'] = env.initial_recon_asp
        env.dynamic_asp[vnf_index]['dyn_recon_counter'] = 0
    else: # it's a CNF
        observation['mtd_action'][cnf_index][0] = 2
        observation['mtd_constraint'][cnf_index][0] -= 1  # reduce migrate limit counter
        # only reset the asp of DoS for CNFs
        env.dynamic_asp[cnf_index]['recon'] = env.initial_recon_asp
        env.dynamic_asp[cnf_index]['dyn_recon_counter'] = 0

    # add reward to security assessment to consider kicking out undetected threats
    reward['proactive_security_reward'] += float(hard_action_proactive_reward)


def update_mtd_constraints(observation, nb_migrations, nb_reinstantiations, nb_stateful_migrations):
    for i in range(observation['nb_resources'][0]):
        if observation['id'][i][0] < 100: # it's a VNF
            observation['mtd_constraint'][i][0] = nb_migrations
            observation['mtd_constraint'][i][1] = nb_reinstantiations
        else: # it's a CNF
            observation['mtd_constraint'][i][0] = nb_stateful_migrations


def is_mtd_budget_zero(observation):
    # print("mtd remaining budget is ", observation['mtd_constraint'])
    for i in range(observation['nb_resources'][0]):
        if observation['mtd_constraint'][i][0] > 0 or observation['mtd_constraint'][i][1] > 0:
            return False
    return True


def mtd_resource_one_row(env, obs, vnf_index):
    # compute the mtd resource overhead based on formula $= coeffcpu * cpu + coeffram * ram_gb + coeffdisk * disk_gb
    coeff_cpu = env.coeff_cpu
    coeff_ram = env.coeff_ram
    coeff_disk = env.coeff_disk
    return (coeff_cpu * obs['resource_consumption'][vnf_index][0]) + (coeff_ram * (obs['resource_consumption'][vnf_index][1] / 1000)) + (
                coeff_disk * obs['resource_consumption'][vnf_index][2])


def mtd_resource_overhead(env,obs):
    total_cost = 0
    for vnf_index in range(0, obs['nb_resources'][0]):
        if obs['mtd_action'][vnf_index][0] != 0:
            obs['mtd_resource_overhead'][vnf_index][0] = mtd_resource_one_row(env, obs, vnf_index)
            total_cost += obs['mtd_resource_overhead'][vnf_index][0]
    return total_cost


def network_overhead(env, obs):
    total_cost = 0
    for vnf_index in range(0, obs['nb_resources'][0]):
        p_loss_rate = obs['network_metrics'][vnf_index][2]
        latency = obs['network_metrics'][vnf_index][0]
        # if p_loss == 0 multiply latency by 0.001
        obs['network_penalty'][vnf_index][0] = (1 + p_loss_rate) * latency
        # if latency is greater than latency_sla multiply the difference by the latency_sla_penalty_factor
        if latency < obs['latency_sla'][vnf_index]:
            total_cost += obs['network_penalty'][vnf_index][0] * latency_sla_penalty_factor
        else:
            total_cost += obs['network_penalty'][vnf_index][0]
    return total_cost


# def proactive_security_assess(env, obs):
#     list_max_values = []
#     for vnf_index in range(0, obs['nb_resources'][0]):
#         # update asp_increase_rates based on asp (MTD effect is already considered in
#         env.dynamic_asp[vnf_index]['recon'] = env.dynamic_asp[vnf_index]['recon'] + np.log(1 + env.dynamic_asp[vnf_index]['recon'] * recon_asp_increase_factor)
#         print('recon asp', env.dynamic_asp[vnf_index]['recon'])
#
#         # if cvss score is 0 we put 0.01 for 0day vulnerabilities
#         max_asp_value = 1.0e+20 + 0.0
#         env.dynamic_asp[vnf_index]['apt'] = min( max_asp_value, max(zero_day_asp, obs['apt_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#         env.dynamic_asp[vnf_index]['dos'] = min( max_asp_value, max(zero_day_asp, obs['dos_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#         env.dynamic_asp[vnf_index]['data_leak'] = min( max_asp_value, max(zero_day_asp, obs['data_leak_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#         env.dynamic_asp[vnf_index]['undefined'] = min( max_asp_value, max(zero_day_asp, obs['undefined_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#
#         # get the max (asp*score) and multiply it by the impact_ssla factor
#         # print(env.dynamic_asp[vnf_index]['apt'], obs['apt_scores'][vnf_index][2])
#         max_vuln = max(env.dynamic_asp[vnf_index]['apt'] * obs['apt_scores'][vnf_index][2],
#                        env.dynamic_asp[vnf_index]['dos'] * obs['dos_scores'][vnf_index][2])
#         max_vuln = max(max_vuln,
#                        env.dynamic_asp[vnf_index]['data_leak'] * obs['data_leak_scores'][vnf_index][2],
#                        env.dynamic_asp[vnf_index]['undefined'] * obs['undefined_scores'][vnf_index][2])
#         large_security_penalty = max_vuln * impact_ssla_factors[obs['impact_ssla'][vnf_index][0]]
#         obs['security_penalty'][vnf_index][0] = np.log(large_security_penalty + 1)
#         # print('For VNF num.',vnf_index,'max vuln', max_vuln, 'security penalty', obs['security_penalty'][vnf_index][0])
#         list_max_values.append(large_security_penalty)
#     # return the sum of all max values
#     return sum(list_max_values)
#

def proactive_security_assess_multiple_null_steps(env, obs, steps):
    list_max_values = []
    for vnf_index in range(0, obs['nb_resources'][0]):
        c = sum(env.recon_schedule[env.dynamic_asp[vnf_index]['dyn_recon_counter']:])
        env.dynamic_asp[vnf_index]['recon'] = c
        env.dynamic_asp[vnf_index]['apt'] = max(zero_day_asp, obs['apt_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']
        env.dynamic_asp[vnf_index]['dos'] = max(zero_day_asp, obs['dos_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']
        env.dynamic_asp[vnf_index]['data_leak'] = max(zero_day_asp, obs['data_leak_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']
        env.dynamic_asp[vnf_index]['undefined'] = max(zero_day_asp, obs['undefined_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']

        # Compute max vulnerability
        max_vuln = max(
            env.dynamic_asp[vnf_index]['apt'] * obs['apt_scores'][vnf_index][2],
            env.dynamic_asp[vnf_index]['dos'] * obs['dos_scores'][vnf_index][2],
            env.dynamic_asp[vnf_index]['data_leak'] * obs['data_leak_scores'][vnf_index][2],
            env.dynamic_asp[vnf_index]['undefined'] * obs['undefined_scores'][vnf_index][2]
        )

        # Compute security penalty
        large_security_penalty = max_vuln * impact_ssla_factors[obs['impact_ssla'][vnf_index][0]]
        obs['security_penalty'][vnf_index][0] = np.log(large_security_penalty + 1)
        list_max_values.append(large_security_penalty)
    # print("the security penalties per VNF are ", list_max_values)
    return sum(list_max_values)


def get_rewards_multiple_null_steps(env, obs, reward_vector, jumped_steps):
    reward_vector['proactive_security_reward'] -= proactive_security_assess_multiple_null_steps(env, obs, jumped_steps)
    # reward['proactive_security_reward'] = - (17.28 * (jumped_steps + 100)) # simulating the results based on previous measurements
    reward_vector['resource_reward'] += reward_vector["resource_reward"] * jumped_steps
    reward_vector['network_reward'] += reward_vector["network_reward"] * jumped_steps


def proactive_security_new(env, obs):
    list_max_values = []
    for vnf_index in range(0, obs['nb_resources'][0]):
        if env.dynamic_asp[vnf_index]['dyn_recon_counter'] < env.recon_T:
            env.dynamic_asp[vnf_index]['recon'] = env.recon_schedule[env.dynamic_asp[vnf_index]['dyn_recon_counter']]
            # print("VNF", vnf_index," has the recon counter at ",env.dynamic_asp[vnf_index]['dyn_recon_counter'] , " and the schedule value is ", env.recon_schedule[env.dynamic_asp[vnf_index]['dyn_recon_counter']])
            # if cvss score is 0 we put 0.01 for 0day vulnerabilities
            env.dynamic_asp[vnf_index]['apt'] = max(zero_day_asp, obs['apt_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']
            env.dynamic_asp[vnf_index]['dos'] = max(zero_day_asp, obs['dos_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']
            env.dynamic_asp[vnf_index]['data_leak'] = max(zero_day_asp, obs['data_leak_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']
            env.dynamic_asp[vnf_index]['undefined'] = max(zero_day_asp, obs['undefined_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon']


        # Compute max vulnerability
        max_vuln = max(
            env.dynamic_asp[vnf_index]['apt'] * obs['apt_scores'][vnf_index][2],
            env.dynamic_asp[vnf_index]['dos'] * obs['dos_scores'][vnf_index][2],
            env.dynamic_asp[vnf_index]['data_leak'] * obs['data_leak_scores'][vnf_index][2],
            env.dynamic_asp[vnf_index]['undefined'] * obs['undefined_scores'][vnf_index][2]
        )

        large_security_penalty = max(max_vuln * impact_ssla_factors[obs['impact_ssla'][vnf_index][0]], zero_day_asp)
        obs['security_penalty'][vnf_index][0] = large_security_penalty
        list_max_values.append(large_security_penalty)
    # print("the security penalties per VNF are ", list_max_values)
    return sum(list_max_values)


# def proactive_security_assess_OLD(env, obs):
#     list_max_values = []
#     for vnf_index in range(0, obs['nb_resources'][0]):
#         # update asp_increase_rates based on asp (MTD effect is already considered in
#         env.dynamic_asp[vnf_index]['recon'] = env.dynamic_asp[vnf_index]['recon'] * (1 + env.dynamic_asp[vnf_index]['recon'])
#         # print('recon asp', env.dynamic_asp[vnf_index]['recon'])
#
#         # if cvss score is 0 we put 0.01 for 0day vulnerabilities
#         max_asp_value = 1.0e+50 + 0.0
#         env.dynamic_asp[vnf_index]['apt'] = min( max_asp_value, max(zero_day_asp, obs['apt_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#         env.dynamic_asp[vnf_index]['dos'] = min( max_asp_value, max(zero_day_asp, obs['dos_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#         env.dynamic_asp[vnf_index]['data_leak'] = min( max_asp_value, max(zero_day_asp, obs['data_leak_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#         env.dynamic_asp[vnf_index]['undefined'] = min( max_asp_value, max(zero_day_asp, obs['undefined_scores'][vnf_index][6]) * env.dynamic_asp[vnf_index]['recon'])
#
#         # get the max (asp*score) and multiply it by the impact_ssla factor
#         # print(env.dynamic_asp[vnf_index]['apt'], obs['apt_scores'][vnf_index][2])
#         max_vuln = max(env.dynamic_asp[vnf_index]['apt'] * obs['apt_scores'][vnf_index][2],
#                        env.dynamic_asp[vnf_index]['dos'] * obs['dos_scores'][vnf_index][2])
#         max_vuln = max(max_vuln,
#                        env.dynamic_asp[vnf_index]['data_leak'] * obs['data_leak_scores'][vnf_index][2],
#                        env.dynamic_asp[vnf_index]['undefined'] * obs['undefined_scores'][vnf_index][2])
#         large_security_penalty = max_vuln * impact_ssla_factors[obs['impact_ssla'][vnf_index][0]]
#         obs['security_penalty'][vnf_index][0] = np.log(large_security_penalty + 1)
#         # print('For VNF num.',vnf_index,'max vuln', max_vuln, 'security penalty', obs['security_penalty'][vnf_index][0])
#         list_max_values.append(large_security_penalty)
#     # return the sum of all max values
#     return sum(list_max_values)

def seconds_to_hours(seconds):
    return seconds / 3600

def get_rewards(env, obs, reward):
    # resource penalty
    mtd_resource_cost = mtd_resource_overhead(env, obs) # * seconds_to_hours(one_step_duration)
    resource_penalty = np.log(mtd_resource_cost + 1)

    # network penalty
    network_penalty = np.log(network_overhead(env, obs) + 1)

    # proactive security assess penalty
    proactive_penalty = proactive_security_new(env, obs)

    # update raward
    reward['resource_reward'] -= resource_penalty
    reward['network_reward'] -= network_penalty
    reward['proactive_security_reward'] -= proactive_penalty