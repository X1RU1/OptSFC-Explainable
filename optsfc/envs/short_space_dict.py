import numpy as np
from gym import spaces
import itertools



cnfs_size = 3
vnfs_size = 4
vims_size = 2

# measure the resouce cost of the operation in near real-time and aggregate it to previous results of the same tuple (action, resource_type / size_unit) to have a better mean
#       the resource cost unit is $, determined by formula $=intercept + coeffcpu * cpu + coeffram * ram_gb + coeffdisk * disk_gb
intercept = -0.0820414
coeff_cpu = 0.03147484
coeff_ram = 0.00424486
coeff_disk = 0.000066249

env_dictionary = {'nb_resources': spaces.Box(low=0, high=vnfs_size + cnfs_size, shape=(1,), dtype=np.uint8),
                    'nb_vims': spaces.Box(low=0, high=vims_size-1, shape=(1,), dtype=np.uint8),
                    'vim_resources': spaces.Box(low=0, high=10000, shape=(vims_size, 3), dtype=np.float16),
                    'id': spaces.Box(low=0, high=100, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'state': spaces.Box(low=0, high=2, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'attack_type': spaces.Box(low=0, high=5, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'vuln_ports_count': spaces.Box(low=0, high=100, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'apt_scores': spaces.Box(low=0, high=1000000, shape=(vnfs_size + cnfs_size,8), dtype=np.float32),
                    'data_leak_scores': spaces.Box(low=-1000000, high=1000000, shape=(vnfs_size + cnfs_size,8), dtype=np.float32),
                    'dos_scores': spaces.Box(low=-1000000, high=1000000, shape=(vnfs_size + cnfs_size,8), dtype=np.float32),
                    'undefined_scores': spaces.Box(low=-1000000, high=1000000, shape=(vnfs_size + cnfs_size,8), dtype=np.float32),
                    'resource_consumption': spaces.Box(low=0, high=10000, shape=(vnfs_size + cnfs_size,3), dtype=np.float16),
                    'mtd_resource_overhead': spaces.Box(low=0, high=60, shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                    'network_penalty': spaces.Box(low=0, high=1, shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                    'security_penalty': spaces.Box(low=0, high=50, shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                    'nb_UEs_cnx': spaces.Box(low=0, high=1000, shape=(vnfs_size + cnfs_size,1), dtype=np.uint16),
                    'vim_host':spaces.Box(low=0, high=vims_size - 1, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'location': spaces.Box(low=0, high=1, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'network_metrics': spaces.Box(low=-1, high=999999999, shape=(vnfs_size + cnfs_size,5), dtype=np.float64),
                    'latency_sla': spaces.Box(low=0, high=1, shape=(vnfs_size + cnfs_size,1), dtype=np.float32),
                    'impact_ssla': spaces.Box(low=0, high=3, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'vnf_parent': spaces.Box(low=0, high=vnfs_size + cnfs_size, shape=(vnfs_size + cnfs_size,1), dtype=np.uint8),
                    'ns_parents': spaces.Box(low=0, high=100, shape=(vnfs_size + cnfs_size,4), dtype=np.uint8),
                    'nsi_parents': spaces.Box(low=0, high=100, shape=(vnfs_size + cnfs_size,4), dtype=np.uint8),
                    'mtd_action': spaces.Box(low=-20, high=255, shape=(vnfs_size + cnfs_size,2), dtype=np.uint8), # nothing, restart, or migrate + duration of the action in seconds
                    'mtd_constraint': spaces.Box(low=0, high=1000, shape=(vnfs_size + cnfs_size,2), dtype=np.uint16)} # remaining migrations + remaining reinst.

observation_dictionary = {'nb_resources': spaces.Box(low=0, high=vnfs_size + cnfs_size, shape=(1,), dtype=np.uint8),
                    'vim_resources': spaces.Box(low=0, high=10000, shape=(vims_size, 3), dtype=np.float16),
                    'id': spaces.Box(low=0, high=100, shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                    'apt_scores': spaces.Box(low=0, high=10, shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                    'data_leak_scores': spaces.Box(low=0, high=10, shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                    'dos_scores': spaces.Box(low=0, high=10, shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                    'undefined_scores': spaces.Box(low=0, high=10, shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                    'mtd_resource_overhead': spaces.Box(low=0, high=60, shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                    'network_penalty': spaces.Box(low=0, high=1, shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                    'security_penalty': spaces.Box(low=0, high=50, shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                    'nb_UEs_cnx': spaces.Box(low=0, high=1000, shape=(vnfs_size + cnfs_size, 1), dtype=np.uint16),
                    'vim_host': spaces.Box(low=0, high=vims_size - 1, shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                    'mtd_action': spaces.Box(low=-20, high=255, shape=(vnfs_size + cnfs_size, 2), dtype=np.uint8), # {nothing, restart, or migrate} + duration of the action in seconds
                    'mtd_constraint': spaces.Box(low=0, high=1000, shape=(vnfs_size + cnfs_size, 2), dtype=np.uint16)}  # remaining migrations + remaining reinst.

space_set_zeros = {'nb_resources': np.zeros(shape=(1,), dtype=np.uint8),
                   'nb_vims': np.zeros(shape=(1,), dtype=np.uint8),
                   'vim_resources': np.zeros(shape=(vims_size, 3), dtype=np.float16),
                   'id': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'state': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'attack_type': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'vuln_ports_count': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'apt_scores': np.zeros(shape=(vnfs_size + cnfs_size, 8), dtype=np.float32),
                   'data_leak_scores': np.zeros(shape=(vnfs_size + cnfs_size, 8), dtype=np.float32),
                   'dos_scores': np.zeros(shape=(vnfs_size + cnfs_size, 8), dtype=np.float32),
                   'undefined_scores': np.zeros(shape=(vnfs_size + cnfs_size, 8), dtype=np.float32),
                   'resource_consumption': np.zeros(shape=(vnfs_size + cnfs_size, 3), dtype=np.float16),
                   'mtd_resource_overhead': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                   'network_penalty': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                   'security_penalty': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                   'nb_UEs_cnx': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint16),
                   'vim_host': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'location': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'network_metrics': np.zeros(shape=(vnfs_size + cnfs_size, 5), dtype=np.float64),
                   'latency_sla': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float32),
                   'impact_ssla': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'vnf_parent': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'ns_parents': np.zeros(shape=(vnfs_size + cnfs_size, 4), dtype=np.uint8),
                   'nsi_parents': np.zeros(shape=(vnfs_size + cnfs_size, 4), dtype=np.uint8),
                   'mtd_action': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.uint8),
                   'mtd_constraint': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.uint16)}

obs_space_set_zeros = {'nb_resources': np.zeros(shape=(1,), dtype=np.uint8),
                   'vim_resources': np.zeros(shape=(vims_size, 3), dtype=np.float16),
                   'id': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'apt_scores': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                   'data_leak_scores': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                   'dos_scores': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                   'undefined_scores': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.float16),
                   'mtd_resource_overhead': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                   'network_penalty': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                   'security_penalty': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.float16),
                   'nb_UEs_cnx': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint16),
                   'vim_host': np.zeros(shape=(vnfs_size + cnfs_size, 1), dtype=np.uint8),
                   'mtd_action': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.uint8),
                   'mtd_constraint': np.zeros(shape=(vnfs_size + cnfs_size, 2), dtype=np.uint16)}

reward_init = {'resource_reward': 0,
               'network_reward': 0,
               'proactive_security_reward': 0}

# STARTING NETWORK SETUP
locations = {'core': 0, 'edge': 1}
# vnf0 has strict latency_ssla
vnf0 = {'id': 4, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 1,
        'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0,
        'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 4.4,
        'data_leak_cvss_score_max': 4.4, 'data_leak_cvss_score_avg': 4.4, 'data_leak_cvss_score_std': 0.0,
        'data_leak_cvss_asp_min': 2.9589945, 'data_leak_cvss_asp_max': 2.9589945, 'data_leak_cvss_asp_avg': 2.9589945,
        'data_leak_cvss_asp_std': 0.0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0,
        'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0,
        'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0,
        'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0,
        'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 1',
        'vim_cpus': 26, 'vim_ram_gb': 196.666, 'vim_disk_gb': 2213, 'vim_location': 'core', 'cpu_cons': 2,
        'ram_cons_gb': 2.048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0,
        'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 3, 'vnf_parent': 'VNF 3',
        'ns_parent1': 'NS 2', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1',
        'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0,
        'incremental_counter': 1}

# vnf1 has great impact_ssla
vnf1 = {'id': 7, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 0,
        'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0,
        'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 0,
        'data_leak_cvss_score_max': 0, 'data_leak_cvss_score_avg': 0, 'data_leak_cvss_score_std': 0,
        'data_leak_cvss_asp_min': 0, 'data_leak_cvss_asp_max': 0, 'data_leak_cvss_asp_avg': 0,
        'data_leak_cvss_asp_std': 0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0,
        'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0,
        'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0,
        'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0,
        'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 2',
        'vim_cpus': 16, 'vim_ram_gb': 96.666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 2,
        'ram_cons_gb': 2.048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0,
        'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 0, 'vnf_parent': 'VNF 6',
        'ns_parent1': 'NS 5', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1',
        'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0,
        'incremental_counter': 1}

vnf2 = {'id': 13, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 0,
        'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0,
        'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 0,
        'data_leak_cvss_score_max': 0, 'data_leak_cvss_score_avg': 0, 'data_leak_cvss_score_std': 0,
        'data_leak_cvss_asp_min': 0, 'data_leak_cvss_asp_max': 0, 'data_leak_cvss_asp_avg': 0,
        'data_leak_cvss_asp_std': 0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0,
        'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0,
        'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0,
        'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0,
        'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 2',
        'vim_cpus': 16, 'vim_ram_gb': 96.666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 2,
        'ram_cons_gb': 2.048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0,
        'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 0, 'vnf_parent': 'VNF 9',
        'ns_parent1': 'NS 8', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1',
        'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0,
        'incremental_counter': 1}

# vnf3 has apt vulnerabilities
vnf3 = {'id': 10, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 2, 'vuln_ports_count': 2,
        'apt_cvss_score_max': 6.5, 'apt_cvss_score_avg': 5.5, 'apt_cvss_score_std': 2, 'apt_cvss_asp_min': 4,
        'apt_cvss_asp_max': 8, 'apt_cvss_asp_avg': 6.5, 'apt_cvss_asp_std': 0.6, 'data_leak_cvss_score_min': 4.3,
        'data_leak_cvss_score_max': 8.8, 'data_leak_cvss_score_avg': 6.957142857142856,
        'data_leak_cvss_score_std': 2.02819963267919, 'data_leak_cvss_asp_min': 2.0680681560000003,
        'data_leak_cvss_asp_max': 2.9589945, 'data_leak_cvss_asp_avg': 2.6589255577499995,
        'data_leak_cvss_asp_std': 0.34346145952646673, 'dos_cvss_score_min': None, 'dos_cvss_score_max': None,
        'dos_cvss_score_avg': None, 'dos_cvss_score_std': None, 'dos_cvss_asp_min': None, 'dos_cvss_asp_max': None,
        'dos_cvss_asp_avg': None, 'dos_cvss_asp_std': None, 'undefined_cvss_score_min': None,
        'undefined_cvss_score_max': None, 'undefined_cvss_score_avg': None, 'undefined_cvss_score_std': None,
        'undefined_cvss_asp_min': None, 'undefined_cvss_asp_max': None, 'undefined_cvss_asp_avg': None,
        'undefined_cvss_asp_std': None, 'vim_host': 'VIM 2', 'vim_cpus': 16, 'vim_ram_gb': 96.666, 'vim_disk_gb': 2013,
        'vim_location': 'edge', 'cpu_cons': 4, 'ram_cons_gb': 8.096, 'disk_cons': 30, 'nb_UEs_cnx': None, 'latency': 0,
        'throughput': 0, 'packet_loss_rate': 0, 'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 0,
        'vnf_parent': 'VNF 12', 'ns_parent1': 'NS 11', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None,
        'nsi_parent1': 'NSi 1', 'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None,
        'network_overhead_cumul_avg': 0.0, 'incremental_counter': 1}


# CNFs (their IDs are >= 100):
cnf0 = {'id': 100, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 1,
        'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0,
        'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 4.4,
        'data_leak_cvss_score_max': 4.4, 'data_leak_cvss_score_avg': 4.4, 'data_leak_cvss_score_std': 0.0,
        'data_leak_cvss_asp_min': 2.9589945, 'data_leak_cvss_asp_max': 2.9589945, 'data_leak_cvss_asp_avg': 2.9589945,
        'data_leak_cvss_asp_std': 0.0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0,
        'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0,
        'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0,
        'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0,
        'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 1',
        'vim_cpus': 26, 'vim_ram_gb': 196.666, 'vim_disk_gb': 2213, 'vim_location': 'core', 'cpu_cons': 2,
        'ram_cons_gb': 2.048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0,
        'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 3, 'vnf_parent': 'VNF 3',
        'ns_parent1': 'NS 2', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1',
        'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0,
        'incremental_counter': 1}

# with DoS CVE-2023-44487
cnf1 = {'id': 101, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 0,
        'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0,
        'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 0,
        'data_leak_cvss_score_max': 0, 'data_leak_cvss_score_avg': 0, 'data_leak_cvss_score_std': 0,
        'data_leak_cvss_asp_min': 0, 'data_leak_cvss_asp_max': 0, 'data_leak_cvss_asp_avg': 0,
        'data_leak_cvss_asp_std': 0, 'dos_cvss_score_min': 7.5, 'dos_cvss_score_max': 7.5, 'dos_cvss_score_avg': 7.5,
        'dos_cvss_score_std': 1, 'dos_cvss_asp_min': 3.9, 'dos_cvss_asp_max': 3.9, 'dos_cvss_asp_avg': 3.9,
        'dos_cvss_asp_std': 1, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0,
        'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0,
        'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 2',
        'vim_cpus': 16, 'vim_ram_gb': 96.666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 2,
        'ram_cons_gb': 2.048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0,
        'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 0, 'vnf_parent': 'VNF 9',
        'ns_parent1': 'NS 8', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1',
        'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0,
        'incremental_counter': 1}

cnf2 = {'id': 102, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 2, 'vuln_ports_count': 2,
        'apt_cvss_score_max': 6.5, 'apt_cvss_score_avg': 5.5, 'apt_cvss_score_std': 2, 'apt_cvss_asp_min': 4,
        'apt_cvss_asp_max': 8, 'apt_cvss_asp_avg': 6.5, 'apt_cvss_asp_std': 0.6, 'data_leak_cvss_score_min': 4.3,
        'data_leak_cvss_score_max': 8.8, 'data_leak_cvss_score_avg': 6.957142857142856,
        'data_leak_cvss_score_std': 2.02819963267919, 'data_leak_cvss_asp_min': 2.0680681560000003,
        'data_leak_cvss_asp_max': 2.9589945, 'data_leak_cvss_asp_avg': 2.6589255577499995,
        'data_leak_cvss_asp_std': 0.34346145952646673, 'dos_cvss_score_min': None, 'dos_cvss_score_max': None,
        'dos_cvss_score_avg': None, 'dos_cvss_score_std': None, 'dos_cvss_asp_min': None, 'dos_cvss_asp_max': None,
        'dos_cvss_asp_avg': None, 'dos_cvss_asp_std': None, 'undefined_cvss_score_min': None,
        'undefined_cvss_score_max': None, 'undefined_cvss_score_avg': None, 'undefined_cvss_score_std': None,
        'undefined_cvss_asp_min': None, 'undefined_cvss_asp_max': None, 'undefined_cvss_asp_avg': None,
        'undefined_cvss_asp_std': None, 'vim_host': 'VIM 2', 'vim_cpus': 16, 'vim_ram_gb': 96.666, 'vim_disk_gb': 2013,
        'vim_location': 'edge', 'cpu_cons': 4, 'ram_cons_gb': 8.096, 'disk_cons': 30, 'nb_UEs_cnx': None, 'latency': 0,
        'throughput': 0, 'packet_loss_rate': 0, 'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla': 0,
        'vnf_parent': 'VNF 12', 'ns_parent1': 'NS 11', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None,
        'nsi_parent1': 'NSi 1', 'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None,
        'network_overhead_cumul_avg': 0.0, 'incremental_counter': 1}

vnfs_list = [vnf0, vnf1, vnf2, vnf3]
cnfs_list = [cnf0, cnf1, cnf2]


# convert all None in vnfs in vnfs_list in to 0
for vnf in vnfs_list:
    for key in vnf:
        if vnf[key] is None:
            vnf[key] = 0
# convert all None in cnfs in cnfs_list in to 0
for cnf in cnfs_list:
    for key in cnf:
        if cnf[key] is None:
            cnf[key] = 0


# Initialize the environment
def init_network_setup(gym_env, environment):
    """" For the first observation we simulate one slice with 4 vnfs (our testbed)
         Set SLAs and resource requirements
    """
    environment['nb_resources'][0] = len(vnfs_list) + len(cnfs_list)
    environment['nb_vims'][0] = vims_size
    for i, vnf in enumerate(itertools.chain(vnfs_list, cnfs_list)):
        environment['id'][i][0] = vnf['id']
        # set variable to 2 if vnf is under attack
        if vnf['state'] == 'ordinary':
            environment['state'][i][0] = 0
        elif vnf['state'] == 'suspicious':
            environment['state'][i][0] = 1
        elif vnf['state'] == 'attack':
            environment['state'][i][0] = 2
        # VIM attributes
        environment['vim_host'][i][0] = int(vnf['vim_host'].split('VIM ')[1]) - 1
        environment['location'][i][0] = locations[vnf['vim_location']]
        environment['vim_resources'][environment['vim_host'][i][0]][0] = vnf['vim_cpus']
        environment['vim_resources'][environment['vim_host'][i][0]][1] = vnf['vim_ram_gb']
        environment['vim_resources'][environment['vim_host'][i][0]][2] = vnf['vim_disk_gb']

        # added this mainly for the simulation of a vulnerable vnf
        environment['apt_scores'][i][0] = vnf['apt_cvss_score_min']
        environment['apt_scores'][i][1] = vnf['apt_cvss_score_max']
        environment['apt_scores'][i][2] = vnf['apt_cvss_score_avg']
        environment['apt_scores'][i][3] = vnf['apt_cvss_score_std']
        environment['apt_scores'][i][4] = vnf['apt_cvss_asp_min']
        environment['apt_scores'][i][5] = vnf['apt_cvss_asp_max']
        environment['apt_scores'][i][6] = vnf['apt_cvss_asp_avg']
        environment['apt_scores'][i][7] = vnf['apt_cvss_asp_std']

        environment['data_leak_scores'][i][0] = vnf['data_leak_cvss_score_min']
        environment['data_leak_scores'][i][1] = vnf['data_leak_cvss_score_max']
        environment['data_leak_scores'][i][2] = vnf['data_leak_cvss_score_avg']
        environment['data_leak_scores'][i][3] = vnf['data_leak_cvss_score_std']
        environment['data_leak_scores'][i][4] = vnf['data_leak_cvss_asp_min']
        environment['data_leak_scores'][i][5] = vnf['data_leak_cvss_asp_max']
        environment['data_leak_scores'][i][6] = vnf['data_leak_cvss_asp_avg']
        environment['data_leak_scores'][i][7] = vnf['data_leak_cvss_asp_std']

        environment['dos_scores'][i][0] = vnf['dos_cvss_score_min']
        environment['dos_scores'][i][1] = vnf['dos_cvss_score_max']
        environment['dos_scores'][i][2] = vnf['dos_cvss_score_avg']
        environment['dos_scores'][i][3] = vnf['dos_cvss_score_std']
        environment['dos_scores'][i][4] = vnf['dos_cvss_asp_min']
        environment['dos_scores'][i][5] = vnf['dos_cvss_asp_max']
        environment['dos_scores'][i][6] = vnf['dos_cvss_asp_avg']
        environment['dos_scores'][i][7] = vnf['dos_cvss_asp_std']

        environment['undefined_scores'][i][0] = vnf['undefined_cvss_score_min']
        environment['undefined_scores'][i][1] = vnf['undefined_cvss_score_max']
        environment['undefined_scores'][i][2] = vnf['undefined_cvss_score_avg']
        environment['undefined_scores'][i][3] = vnf['undefined_cvss_score_std']
        environment['undefined_scores'][i][4] = vnf['undefined_cvss_asp_min']
        environment['undefined_scores'][i][5] = vnf['undefined_cvss_asp_max']
        environment['undefined_scores'][i][6] = vnf['undefined_cvss_asp_avg']
        environment['undefined_scores'][i][7] = vnf['undefined_cvss_asp_std']

        environment['resource_consumption'][i][0] = vnf['cpu_cons']
        environment['resource_consumption'][i][1] = vnf['ram_cons_gb']
        environment['resource_consumption'][i][2] = vnf['disk_cons']
        environment['vnf_parent'][i][0] = int(vnf['vnf_parent'].split('VNF ')[1])
        environment['ns_parents'][i][0] = 0 if vnf['ns_parent1'] == 0 else int(vnf['ns_parent1'].split('NS ')[1])
        environment['ns_parents'][i][1] = 0 if vnf['ns_parent2'] == 0 else int(vnf['ns_parent2'].split('NS ')[1])
        environment['ns_parents'][i][2] = 0 if vnf['ns_parent3'] == 0 else int(vnf['ns_parent3'].split('NS ')[1])
        environment['ns_parents'][i][3] = 0 if vnf['ns_parent4'] == 0 else int(vnf['ns_parent4'].split('NS ')[1])
        environment['nsi_parents'][i][0] = 0 if vnf['nsi_parent1'] == 0 else int(vnf['nsi_parent1'].split('NSi ')[1])
        environment['nsi_parents'][i][1] = 0 if vnf['nsi_parent2'] == 0 else int(vnf['nsi_parent2'].split('NSi ')[1])
        environment['nsi_parents'][i][2] = 0 if vnf['nsi_parent3'] == 0 else int(vnf['nsi_parent3'].split('NSi ')[1])
        environment['nsi_parents'][i][3] = 0 if vnf['nsi_parent4'] == 0 else int(vnf['nsi_parent4'].split('NSi ')[1])
        environment['nb_UEs_cnx'][i][0] = 4
        environment['impact_ssla'][i][0] = vnf['impact_ssla']
        if i == 0:
            environment['latency_sla'][i][0] = 0.0165 # black sheep to see if it stays in the edge
            environment['nb_UEs_cnx'][i][0] = 10
        else:
            environment['latency_sla'][i][0] = 0.05
        if vnf['id'] < 100:
            environment['mtd_constraint'][i][0] = gym_env.migrations_per_month
            environment['mtd_constraint'][i][1] = gym_env.reinstantiations_per_month
        else:
            environment['mtd_constraint'][i][0] = gym_env.stateful_migrations_per_month
            environment['mtd_constraint'][i][1] = 0
    return environment

def update_agent_obs(environment, observation):
    observation['nb_resources'][0] = environment['nb_resources'][0]
    for i in range(0, environment['nb_resources'][0]):
        observation['id'][i][0] = environment['id'][i][0]
        observation['vim_host'][i][0] = environment['vim_host'][i][0]
        observation['nb_UEs_cnx'][i][0] = environment['nb_UEs_cnx'][i][0]

        # added this mainly for the simulation of a vulnerable vnf
        observation['apt_scores'][i][0] = environment['apt_scores'][i][2]
        observation['apt_scores'][i][1] = environment['apt_scores'][i][6]
        observation['data_leak_scores'][i][0] = environment['data_leak_scores'][i][2]
        observation['data_leak_scores'][i][1] = environment['data_leak_scores'][i][6]
        observation['dos_scores'][i][0] = environment['dos_scores'][i][2]
        observation['dos_scores'][i][1] = environment['dos_scores'][i][6]
        observation['undefined_scores'][i][0] = environment['undefined_scores'][i][2]
        observation['undefined_scores'][i][1] = environment['undefined_scores'][i][6]
        observation['mtd_constraint'][i][0] = environment['mtd_constraint'][i][0]
        observation['mtd_constraint'][i][1] = environment['mtd_constraint'][i][1]
        observation['mtd_action'][i] = environment['mtd_action'][i]

        observation['mtd_resource_overhead'][i][0] = environment['mtd_resource_overhead'][i][0]
        observation['network_penalty'][i][0] = environment['network_penalty'][i][0]
        observation['security_penalty'][i][0] = environment['security_penalty'][i][0]

    # attributes per VIM
    for i in range(0, environment['nb_vims'][0]):
        # get remaining resources
        observation['vim_resources'][i][0] = environment['vim_resources'][i][0]
        observation['vim_resources'][i][1] = environment['vim_resources'][i][1]
        observation['vim_resources'][i][2] = environment['vim_resources'][i][2]
        # print("vim_resources", observation['vim_resources'][i][0], observation['vim_resources'][i][1], observation['vim_resources'][i][2])
    return observation

def space_init(gym_env):
    return init_network_setup(gym_env, space_set_zeros)

def obs_init(gym_env):
    return update_agent_obs(init_network_setup(gym_env, space_set_zeros), obs_space_set_zeros)