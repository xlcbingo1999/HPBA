current_ip_index = 5
current_cmd_index = 7

is_simulation = True
need_save_jobtrace_flag = False
worker_indexes = [3]
worker_indexes = [str(index) for index in worker_indexes]
worker_indexes_str = " ".join(worker_indexes)

assignment_policy = "HISPolicy"
his_batch_size_for_one_epochs = 100
significance_policy = "TempPolicy"
test_jobtrace_reconstruct_path = "schedule-review-simulation-05-09-21-11-48" # "schedule-review-simulation-05-04-00-43-38"
dataset_reconstruct_path = "schedule-review-simulation-05-09-21-11-48" # "schedule-review-simulation-05-03-19-49-14"
history_jobtrace_reconstruct_path = "schedule-review-simulation-05-09-21-11-48" # "schedule-review-simulation-05-03-19-49-14"
all_decision_num = 1000
datablock_require_epsilon_max_ratio = 0.1
change_job_epsilon_max_times = 1.0
all_history_num = 0
his_betas = 0.0
simulation_all_datablock_num = 100
simulation_offline_datablock_num = 100
change_datablock_epsilon_max_times = 1.0
simulation_time = 5
waiting_time = 2 if is_simulation else 10
seeds = [1234, 2345, 3456, 6789, 7890] if is_simulation else [1234]
seeds = [str(seed) for seed in seeds]
seed_str = " ".join(seeds)

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

dispatcher_cmds.append(f"--all_decision_num {all_decision_num}")
dispatcher_cmds.append(f"--datablock_require_epsilon_max_ratio {datablock_require_epsilon_max_ratio}")
dispatcher_cmds.append(f"--change_job_epsilon_max_times {change_job_epsilon_max_times}")
dispatcher_cmds.append(f"--change_datablock_epsilon_max_times {change_datablock_epsilon_max_times}")
dispatcher_cmds.append(f"--all_history_num {all_history_num}")
dispatcher_cmds.append(f"--his_betas {his_betas}")

if is_simulation:
    dispatcher_cmds.append(f"--simulation_flag")
    dispatcher_cmds.append(f"--simulation_all_datablock_num {simulation_all_datablock_num}")
    dispatcher_cmds.append(f"--simulation_offline_datablock_num {simulation_offline_datablock_num}")
    dispatcher_cmds.append(f"--simulation_time {simulation_time}")
else:
    dispatcher_cmds.append(f"--worker_indexes {worker_indexes_str}")

if need_save_jobtrace_flag:
    dispatcher_cmds.append(f"--need_save_jobtrace_flag")

dispatcher_cmds.append(f"--seed {seed_str}")
dispatcher_cmds.append(f"--waiting_time {waiting_time}")
print(" ".join(dispatcher_cmds))
print("======= =======")