# 合理的Trace: 默认任务到达间隔是 12s / 个 
# 合理的Trace: 默认块的到达间隔是 120s / 个 (10倍)
# 如果把时间进行压缩, 则应该修改到达速率
# [4x] => 3s/个 30s/个 => 200个任务可以控制在1h, 20个online block

current_ip_index = 2
current_cmd_index = 1

# testbed
worker_indexes = [2, 3]
worker_indexes = [str(index) for index in worker_indexes]
worker_indexes_str = " ".join(worker_indexes)
# simulation
simulation_flag = False
simulation_time = 5

# 数据集
test_jobtrace_reconstruct_path = "schedule-review-testbed-06-07-12-53-43"
dataset_reconstruct_path = "schedule-review-testbed-06-07-12-53-43"
history_jobtrace_reconstruct_path = "schedule-review-testbed-06-07-12-53-43"
need_save_jobtrace_flag = False

# 全局设置
max_gpu_fuzai = 10
all_or_nothing_flag = True
enable_waiting_flag = False
seeds = [1234, 2345, 3456, 6789, 7890] if simulation_flag else [1234]
seeds = [str(seed) for seed in seeds]
seed_str = " ".join(seeds)
waiting_time = 2 if simulation_flag else 10

# 任务
pipeline_sequence_all_num = 200
all_history_num = 200
job_arrival_time_speed_up = 4.0 # 控制到达速率
job_datablock_epsilon_max_ratio = 0.2 # 这个控制比率(离群值控制)
change_job_epsilon_max_times = 1.0 # 这个直接从平均增大倍数(平均值控制)
job_require_select_block_min_num = 2
job_require_select_block_max_num = 4
config_max_operate_siton_run_num = 1

# block
all_datablock_num = 20
offline_datablock_num = 20
datablock_arrival_time_speed_up = 4.0 # 控制到达速率
base_capacity = 5.0
dataset_name = "EMNIST"
dataset_config_name = "subtrain_144_split_1.0_dirichlet"

assignment_policy = "BestFitwithRemainPolicy"
his_betas = 0.0
his_batch_size_for_one_epochs = 5
his_infinity_flag = True
pbg_comparison_cost_epsilons = 0.0
pbg_comparison_z_thresholds = 0.9
pbg_Ls = 0.01
pbg_Us = 0.5
pbg_gittas = 0.1

significance_policy = "OTDDPolicy"
temp_sig_metric = "Accuracy"

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
if simulation_flag:
    dispatcher_cmds.append(f"--simulation_flag")
    dispatcher_cmds.append(f"--simulation_time {simulation_time}")
else:
    dispatcher_cmds.append(f"--worker_indexes {worker_indexes_str}")
    

dispatcher_cmds.append(f"--sched_ip 172.18.162.{current_ip_index}")
dispatcher_cmds.append(f"--sched_port 163{current_cmd_index}{current_ip_index}")
dispatcher_cmds.append(f"--dispatcher_ip 172.18.162.{current_ip_index}")
dispatcher_cmds.append(f"--dispatcher_port 164{current_cmd_index}{current_ip_index}")

if len(test_jobtrace_reconstruct_path) > 0:
    dispatcher_cmds.append(f"--test_jobtrace_reconstruct_path {test_jobtrace_reconstruct_path}")
if len(dataset_reconstruct_path) > 0:
    dispatcher_cmds.append(f"--dataset_reconstruct_path {dataset_reconstruct_path}")
if len(history_jobtrace_reconstruct_path) > 0:
    dispatcher_cmds.append(f"--history_jobtrace_reconstruct_path {history_jobtrace_reconstruct_path}")

# 全局
dispatcher_cmds.append(f"--max_gpu_fuzai {max_gpu_fuzai}")
if all_or_nothing_flag:
    dispatcher_cmds.append(f"--all_or_nothing_flag")
if enable_waiting_flag:
    dispatcher_cmds.append(f"--enable_waiting_flag")
if need_save_jobtrace_flag:
    dispatcher_cmds.append(f"--need_save_jobtrace_flag")
dispatcher_cmds.append(f"--seed {seed_str}")
dispatcher_cmds.append(f"--waiting_time {waiting_time}")

# 任务
dispatcher_cmds.append(f"--pipeline_sequence_all_num {pipeline_sequence_all_num}")
dispatcher_cmds.append(f"--all_history_num {all_history_num}")
dispatcher_cmds.append(f"--job_arrival_time_speed_up {job_arrival_time_speed_up}")
dispatcher_cmds.append(f"--job_datablock_epsilon_max_ratio {job_datablock_epsilon_max_ratio}") # 这个控制比率
dispatcher_cmds.append(f"--change_job_epsilon_max_times {change_job_epsilon_max_times}") # 这个直接从平均增大倍数
dispatcher_cmds.append(f"--job_require_select_block_min_num {job_require_select_block_min_num}")
dispatcher_cmds.append(f"--job_require_select_block_max_num {job_require_select_block_max_num}")
dispatcher_cmds.append(f"--config_max_operate_siton_run_num {config_max_operate_siton_run_num}")

# block
dispatcher_cmds.append(f"--all_datablock_num {all_datablock_num}")
dispatcher_cmds.append(f"--offline_datablock_num {offline_datablock_num}")
dispatcher_cmds.append(f"--datablock_arrival_time_speed_up {datablock_arrival_time_speed_up}")
dispatcher_cmds.append(f"--base_capacity {base_capacity}")
dispatcher_cmds.append(f"--dataset_name {dataset_name}")
dispatcher_cmds.append(f"--dataset_config_name {dataset_config_name}")

# 调度决策
dispatcher_cmds.append(f"--assignment_policy {assignment_policy}")
if "PBG" in assignment_policy:
    dispatcher_cmds.append(f"--pbg_comparison_cost_epsilons {pbg_comparison_cost_epsilons}")
    dispatcher_cmds.append(f"--pbg_comparison_z_thresholds {pbg_comparison_z_thresholds}")
    dispatcher_cmds.append(f"--pbg_Ls {pbg_Ls}")
    dispatcher_cmds.append(f"--pbg_Us {pbg_Us}")
    dispatcher_cmds.append(f"--pbg_gittas {pbg_gittas}")

if "HIS" in assignment_policy:
    dispatcher_cmds.append(f"--his_betas {his_betas}")
    dispatcher_cmds.append(f"--his_batch_size_for_one_epochs {his_batch_size_for_one_epochs}")
    if his_infinity_flag:
        dispatcher_cmds.append(f"--his_infinity_flag")

dispatcher_cmds.append(f"--significance_policy {significance_policy}")
dispatcher_cmds.append(f"--temp_sig_metric {temp_sig_metric}")

print(" ".join(dispatcher_cmds))
print("======= =======")