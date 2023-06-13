
import json
all_cmds = "python DL_dispatcher.py --worker_ips 172.18.162.5 --worker_ports 16215 --worker_indexes 2 3 --sched_ip 172.18.162.5 --sched_port 16315 --dispatcher_ip 172.18.162.5 --dispatcher_port 16415 --test_jobtrace_reconstruct_path schedule-review-testbed-06-08-00-42-40 --dataset_reconstruct_path schedule-review-testbed-06-08-00-42-40 --history_jobtrace_reconstruct_path schedule-review-testbed-06-08-00-42-40 --max_gpu_fuzai 5 --seed 1234 --waiting_time 10 --pipeline_sequence_all_num 2 --all_history_num 2 --job_arrival_time_speed_up 4.0 --job_datablock_epsilon_max_ratio 0.2 --change_job_epsilon_max_times 1.0 --job_require_select_block_min_num 4 --job_require_select_block_max_num 4 --config_max_operate_siton_run_num 1 --all_datablock_num 20 --offline_datablock_num 20 --datablock_arrival_time_speed_up 4.0 --base_capacity 5.0 --dataset_name EMNIST --dataset_config_name subtrain_144_split_1.0_dirichlet --assignment_policy HISwithOrderProVersionPolicy --his_betas 0.0 --his_batch_size_for_one_epochs 5 --his_infinity_flag --significance_policy OTDDPolicy --temp_sig_metric Accuracy"

# print(all_cmds)
cmd_sub_str_arr = all_cmds.split(" ")
cmd_sub_str_arr = [s for s in cmd_sub_str_arr if len(s) > 0]
print(json.dumps(cmd_sub_str_arr))