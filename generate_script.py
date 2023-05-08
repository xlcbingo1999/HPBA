current_ip_index = 5
current_cmd_index = 1

assignment_policy = "IterativeHISwithOrderPolicy"
his_batch_size_for_one_epochs = 10
significance_policy = "TempPolicy"
test_jobtrace_reconstruct_path = "schedule-review-simulation-05-04-00-39-24"
dataset_reconstruct_path = "schedule-review-simulation-05-03-19-49-14"
history_jobtrace_reconstruct_path = "schedule-review-simulation-05-03-19-49-14"
all_decision_num = 3000
simulation_datablock_require_epsilon_max_ratio = 0.1
all_history_num = 0
his_betas = 0.0
simulation_all_datablock_num = 100
simulation_offline_datablock_num = 100
simulation_time = 5
seeds = [1234, 2345, 3456, 6789, 7890]
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

dispatcher_cmds.append(f"--test_jobtrace_reconstruct_path {test_jobtrace_reconstruct_path}")
dispatcher_cmds.append(f"--dataset_reconstruct_path {dataset_reconstruct_path}")
dispatcher_cmds.append(f"--history_jobtrace_reconstruct_path {history_jobtrace_reconstruct_path}")

dispatcher_cmds.append(f"--all_decision_num {all_decision_num}")
dispatcher_cmds.append(f"--simulation_datablock_require_epsilon_max_ratio {simulation_datablock_require_epsilon_max_ratio}")
dispatcher_cmds.append(f"--all_history_num {all_history_num}")
dispatcher_cmds.append(f"--his_betas {his_betas}")

dispatcher_cmds.append(f"--simulation_all_datablock_num {simulation_all_datablock_num}")
dispatcher_cmds.append(f"--simulation_offline_datablock_num {simulation_offline_datablock_num}")
dispatcher_cmds.append(f"--simulation_flag")
dispatcher_cmds.append(f"--simulation_time {simulation_time}")
dispatcher_cmds.append(f"--seed {seed_str}")
print(" ".join(dispatcher_cmds))
print("======= =======")
