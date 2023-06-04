current_ip_index = 5
current_cmd_index = 7

is_simulation = False
all_or_nothing_flag = True
enable_waiting_flag = True
pipeline_sequence_all_num = 10

need_save_jobtrace_flag = False
worker_indexes = [0, 1]
worker_indexes = [str(index) for index in worker_indexes]
worker_indexes_str = " ".join(worker_indexes)

assignment_policy = "BestFitwithRemainPolicy"
his_batch_size_for_one_epochs = 5
significance_policy = "OTDDPolicy"
test_jobtrace_reconstruct_path = "schedule-review-testbed-06-03-23-37-06" # "schedule-review-simulation-05-09-21-11-48" # "schedule-review-simulation-05-04-00-43-38"
dataset_reconstruct_path = "schedule-review-testbed-06-03-23-37-06" # "schedule-review-simulation-05-09-21-11-48" # "schedule-review-simulation-05-03-19-49-14"
history_jobtrace_reconstruct_path = "schedule-review-testbed-06-03-23-37-06" # "schedule-review-simulation-05-03-19-49-14"
dataset_name = "EMNIST"
dataset_config_name = "subtrain_144_split_1.0_dirichlet"

datablock_require_epsilon_max_ratio = 0.1
job_require_select_block_min_num = 5
job_require_select_block_max_num = 25
change_job_epsilon_max_times = 1.0
all_history_num = 0
his_betas = 0.0
all_datablock_num = 100
offline_datablock_num = 100
base_capacity = 3.0
change_datablock_epsilon_max_times = 1.0
simulation_time = 5
waiting_time = 2 if is_simulation else 10
seeds = [1234, 2345, 3456, 6789, 7890] if is_simulation else [1234]
seeds = [str(seed) for seed in seeds]
seed_str = " ".join(seeds)

pbg_comparison_cost_epsilons = 0.0
pbg_comparison_z_thresholds = 0.9
pbg_Ls = 0.01
pbg_Us = 10.0
pbg_gittas = 0.1

print("======= worker =======")
worker_cmds = []
worker_cmds.append(f"python DL_worker.py")
worker_cmds.append(f"--local_ip 172.18.162.{current_ip_index}")
worker_cmds.append(f"--local_port 162{current_cmd_index}{current_ip_index}")
worker_cmds.append(f"--sched_ip 172.18.162.{current_ip_index}")
worker_cmds.append(f"--sched_port 163{current_cmd_index}{current_ip_index}")
print(" ".join(worker_cmds))
print("======= =======")

print("======= sched =======")
sched_cmds = []
sched_cmds.append(f"python DL_sched.py")
sched_cmds.append(f"--worker_ips 172.18.162.{current_ip_index}")
sched_cmds.append(f"--worker_ports 162{current_cmd_index}{current_ip_index}")
sched_cmds.append(f"--sched_ip 172.18.162.{current_ip_index}")
sched_cmds.append(f"--sched_port 163{current_cmd_index}{current_ip_index}")
print(" ".join(sched_cmds))
print("======= =======")

print("======= dispatcher =======")
dispatcher_cmds = []
dispatcher_cmds.append(f"python DL_dispatcher.py")
dispatcher_cmds.append(f"--worker_ips 172.18.162.{current_ip_index}")
dispatcher_cmds.append(f"--worker_ports 162{current_cmd_index}{current_ip_index}")
dispatcher_cmds.append(f"--sched_ip 172.18.162.{current_ip_index}")
dispatcher_cmds.append(f"--sched_port 163{current_cmd_index}{current_ip_index}")
dispatcher_cmds.append(f"--dispatcher_ip 172.18.162.{current_ip_index}")
dispatcher_cmds.append(f"--dispatcher_port 164{current_cmd_index}{current_ip_index}")

dispatcher_cmds.append(f"--assignment_policy {assignment_policy}")
dispatcher_cmds.append(f"--his_batch_size_for_one_epochs {his_batch_size_for_one_epochs}")
dispatcher_cmds.append(f"--significance_policy {significance_policy}")

if len(test_jobtrace_reconstruct_path) > 0:
    dispatcher_cmds.append(f"--test_jobtrace_reconstruct_path {test_jobtrace_reconstruct_path}")
if len(dataset_reconstruct_path) > 0:
    dispatcher_cmds.append(f"--dataset_reconstruct_path {dataset_reconstruct_path}")
if len(history_jobtrace_reconstruct_path) > 0:
    dispatcher_cmds.append(f"--history_jobtrace_reconstruct_path {history_jobtrace_reconstruct_path}")

dispatcher_cmds.append(f"--dataset_name {dataset_name}")
dispatcher_cmds.append(f"--dataset_config_name {dataset_config_name}")

dispatcher_cmds.append(f"--pipeline_sequence_all_num {pipeline_sequence_all_num}")
dispatcher_cmds.append(f"--datablock_require_epsilon_max_ratio {datablock_require_epsilon_max_ratio}")
dispatcher_cmds.append(f"--job_require_select_block_min_num {job_require_select_block_min_num}")
dispatcher_cmds.append(f"--job_require_select_block_max_num {job_require_select_block_max_num}")
dispatcher_cmds.append(f"--change_job_epsilon_max_times {change_job_epsilon_max_times}")
dispatcher_cmds.append(f"--base_capacity {base_capacity}")
dispatcher_cmds.append(f"--change_datablock_epsilon_max_times {change_datablock_epsilon_max_times}")
dispatcher_cmds.append(f"--all_history_num {all_history_num}")
dispatcher_cmds.append(f"--his_betas {his_betas}")
dispatcher_cmds.append(f"--all_datablock_num {all_datablock_num}")
dispatcher_cmds.append(f"--offline_datablock_num {offline_datablock_num}")

if "PBG" in assignment_policy:
    dispatcher_cmds.append(f"--pbg_comparison_cost_epsilons {pbg_comparison_cost_epsilons}")
    dispatcher_cmds.append(f"--pbg_comparison_z_thresholds {pbg_comparison_z_thresholds}")
    dispatcher_cmds.append(f"--pbg_Ls {pbg_Ls}")
    dispatcher_cmds.append(f"--pbg_Us {pbg_Us}")
    dispatcher_cmds.append(f"--pbg_gittas {pbg_gittas}")

if "HIS" in assignment_policy:
    dispatcher_cmds.append(f"--his_batch_size_for_one_epochs {his_batch_size_for_one_epochs}")

if is_simulation:
    dispatcher_cmds.append(f"--simulation_flag")
    dispatcher_cmds.append(f"--simulation_time {simulation_time}")
else:
    dispatcher_cmds.append(f"--worker_indexes {worker_indexes_str}")

if all_or_nothing_flag:
    dispatcher_cmds.append(f"--all_or_nothing_flag")
if enable_waiting_flag:
    dispatcher_cmds.append(f"--enable_waiting_flag")
if need_save_jobtrace_flag:
    dispatcher_cmds.append(f"--need_save_jobtrace_flag")

dispatcher_cmds.append(f"--seed {seed_str}")
dispatcher_cmds.append(f"--waiting_time {waiting_time}")
print(" ".join(dispatcher_cmds))
print("======= =======")