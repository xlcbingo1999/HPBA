# 合理的Trace: 默认任务到达间隔是 12 s
# 合理的Trace: 默认块的到达间隔是 120 s (10倍)
# 如果把时间进行压缩, 则应该修改到达速率
# [4x] => 3 s; 30 s
import os
import json

nohup_flag = False
debug_flag = False
nohup_target_dir_prefix = "/home/netlab/DL_lab/opacus_testbed/log_temp_store/"
target_time_minute = 180

current_ip_index = 5
current_cmd_index = 3

# testbed
worker_indexes = [current_cmd_index]
worker_indexes = [str(index) for index in worker_indexes]
worker_indexes_str = " ".join(worker_indexes)
# simulation
simulation_flag = False
simulation_time = 5

# 数据集
test_jobtrace_reconstruct_path = "schedule-review-testbed-06-18-12-28-24"
dataset_reconstruct_path = "schedule-review-testbed-06-18-12-28-24"
history_jobtrace_reconstruct_path = "schedule-review-testbed-06-18-12-28-24"
need_save_jobtrace_flag = False

# 全局设置
max_gpu_fuzai = 10000000 if simulation_flag else 5 
all_or_nothing_flag = False
enable_waiting_flag = False
inf_job_dispatch_flag = True
need_stop_lower_bound_ratio = 0.1 # 检测结束, 在base_capacity为5.0时默认设置为0.1
seeds = [1234, 2345, 3456, 6789, 7890] if simulation_flag else [3456]
seeds = [str(seed) for seed in seeds]
seed_str = " ".join(seeds)
waiting_time = 2 if simulation_flag else 10

# 任务
pipeline_sequence_all_num = 1000
all_history_num = 1600 # 在INF场景中这个东西太多似乎不太好
job_arrival_time_speed_up = 4.0 # 控制到达速率
job_datablock_epsilon_max_ratio = 0.2 # 控制最大的比率(离群值控制)
job_datablock_epsilon_min_ratio = 0.04 # 控制最小的比率(离群值控制)
job_datablock_lambda = 0.2
change_job_epsilon_max_times = 1.0 * (job_datablock_lambda / job_datablock_epsilon_max_ratio) # 这个直接从平均增大倍数(平均值控制),
                                   # 逻辑1: 先保证任务的请求量在(job_datablock_epsilon_min_ratio, job_datablock_epsilon_max_ratio)之间
                                   # 逻辑2: 然后再直接将任务的请求量做倍数放大或缩小
job_require_select_block_min_num = 4
job_require_select_block_max_num = 4
config_max_operate_siton_run_num = 1

# block
all_datablock_num = 60
offline_datablock_num = 20
datablock_arrival_time_speed_up = 4.0 # 控制到达速率
base_capacity = 5.0
dataset_name = "EMNIST"
dataset_config_name = "subtrain_144_split_1.0_dirichlet"

assignment_policy = "HISwithOrderProVersionPolicy"
his_betas = 0.0
his_batch_size_for_one_epochs = 0
his_infinity_flag = True
his_stop_n_growing_flag = True
his_greedy_flag = False
his_greedy_threshold = 0.2
his_adaptive_cons_generate_flag = "MinBlock" # None, Fair, MinBlock
pbg_comparison_cost_epsilons = 0.0
pbg_comparison_z_thresholds = 0.9
pbg_Ls = 0.01
pbg_Us = 0.5
pbg_gittas = 0.1

significance_policy = "TempPolicy"
temp_sig_metric = "Accuracy"



print("======= worker =======")
worker_cmds = []
worker_cmds.append(f"python DL_worker.py")
worker_cmds.append(f"--local_ip 172.18.162.{current_ip_index}")
worker_cmds.append(f"--local_port 162{current_cmd_index}{current_ip_index}")
worker_cmds.append(f"--sched_ip 172.18.162.{current_ip_index}")
worker_cmds.append(f"--sched_port 163{current_cmd_index}{current_ip_index}")
normal_worker_cmd = " ".join(worker_cmds)
if nohup_flag:
    nohup_worker_dir = os.path.join(nohup_target_dir_prefix, f"worker_{target_time_minute}_{assignment_policy}_{pipeline_sequence_all_num}.log")
    print(f"at now +{target_time_minute} minute")
    print(f"nohup {normal_worker_cmd} > {nohup_worker_dir} 2>&1 &")
elif debug_flag:
    cmd_sub_str_arr = normal_worker_cmd.split(" ")
    cmd_sub_str_arr = [s for s in cmd_sub_str_arr if len(s) > 0][2:]
    print(json.dumps(cmd_sub_str_arr))
else:
    print(normal_worker_cmd)
print("======= =======")

print("======= sched =======")
sched_cmds = []
sched_cmds.append(f"python DL_sched.py")
sched_cmds.append(f"--worker_ips 172.18.162.{current_ip_index}")
sched_cmds.append(f"--worker_ports 162{current_cmd_index}{current_ip_index}")
sched_cmds.append(f"--sched_ip 172.18.162.{current_ip_index}")
sched_cmds.append(f"--sched_port 163{current_cmd_index}{current_ip_index}")
normal_sched_cmd = " ".join(sched_cmds)
if nohup_flag:
    nohup_sched_dir = os.path.join(nohup_target_dir_prefix, f"sched_{target_time_minute}_{assignment_policy}_{pipeline_sequence_all_num}.log")
    print(f"at now +{target_time_minute+2} minute")
    print(f"nohup {normal_sched_cmd} > {nohup_sched_dir} 2>&1 &")
elif debug_flag:
    cmd_sub_str_arr = normal_sched_cmd.split(" ")
    cmd_sub_str_arr = [s for s in cmd_sub_str_arr if len(s) > 0][2:]
    
    print(json.dumps(cmd_sub_str_arr))
else:
    print(normal_sched_cmd)
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
if inf_job_dispatch_flag:
    dispatcher_cmds.append(f"--inf_job_dispatch_flag")
    dispatcher_cmds.append(f"--need_stop_lower_bound_ratio {need_stop_lower_bound_ratio}")
dispatcher_cmds.append(f"--pipeline_sequence_all_num {pipeline_sequence_all_num}")
dispatcher_cmds.append(f"--all_history_num {all_history_num}")
dispatcher_cmds.append(f"--job_arrival_time_speed_up {job_arrival_time_speed_up}")
dispatcher_cmds.append(f"--job_datablock_epsilon_max_ratio {job_datablock_epsilon_max_ratio}") # 这个控制比率
dispatcher_cmds.append(f"--job_datablock_epsilon_min_ratio {job_datablock_epsilon_min_ratio}") # 这个控制比率
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
    dispatcher_cmds.append(f"--his_adaptive_cons_generate_flag {his_adaptive_cons_generate_flag}")
    if his_infinity_flag:
        dispatcher_cmds.append(f"--his_infinity_flag")
    if his_stop_n_growing_flag:
        dispatcher_cmds.append(f"--his_stop_n_growing_flag")
    if his_greedy_flag:
        dispatcher_cmds.append(f"--his_greedy_flag")
        dispatcher_cmds.append(f"--his_greedy_threshold {his_greedy_threshold}")
    else:
        dispatcher_cmds.append(f"--his_greedy_threshold 1.0")

dispatcher_cmds.append(f"--significance_policy {significance_policy}")
dispatcher_cmds.append(f"--temp_sig_metric {temp_sig_metric}")

normal_dispatcher_cmd = " ".join(dispatcher_cmds)
if nohup_flag:
    nohup_dispatcher_dir = os.path.join(nohup_target_dir_prefix, f"dispatcher_{target_time_minute}_{assignment_policy}_{pipeline_sequence_all_num}.log")
    print(f"at now +{target_time_minute+4} minute")
    print(f"nohup {normal_dispatcher_cmd} > {nohup_dispatcher_dir} 2>&1 &")
elif debug_flag:
    cmd_sub_str_arr = normal_dispatcher_cmd.split(" ")
    cmd_sub_str_arr = [s for s in cmd_sub_str_arr if len(s) > 0][2:]
    print(json.dumps(cmd_sub_str_arr))
else:
    print(normal_dispatcher_cmd)
print("======= =======")