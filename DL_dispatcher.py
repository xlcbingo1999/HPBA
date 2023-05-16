import zerorpc
import time
from utils.global_variable import RESULT_PATH, RECONSTRUCT_TRACE_PREFIX_PATH
from utils.global_functions import get_types, convert_types
import threading
from functools import reduce
import sys
import argparse
import json
import copy
import os
import numpy as np
from queue import PriorityQueue
from utils.logging_tools import get_logger
from utils.generate_tools import generate_dataset, generate_jobs, generate_alibaba_jobs, generate_alibaba_dataset
from utils.data_operator import read_DL_dispatcher_result_func

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--dataset_reconstruct_path", type=str, default="")
    parser.add_argument("--test_jobtrace_reconstruct_path", type=str, default="")
    parser.add_argument("--history_jobtrace_reconstruct_path", type=str, default="")
    parser.add_argument("--save_to_origin_trace_path", type=str, default="")

    parser.add_argument("--global_sleep_time", type=int, default=5)
    parser.add_argument("--all_decision_num", type=int, default=50)
    parser.add_argument("--all_history_num", type=int, default=50)
    parser.add_argument("--time_interval", type=int, default=100) # 100, 500, 1000, 1500 [1/1, 1/2, 1/3, 1/5]
    parser.add_argument("--need_change_interval", action="store_true")
    parser.add_argument("--need_save_jobtrace_flag", action="store_true")
    
    
    parser.add_argument("--simulation_flag", action="store_true")
    parser.add_argument("--simulation_time", type=int, default=1)
    parser.add_argument("--simulation_time_speed_up", type=float, default=1.0)
    parser.add_argument("--simulation_all_datablock_num", type=int, default=100)
    parser.add_argument("--simulation_offline_datablock_num", type=int, default=100)

    parser.add_argument("--datablock_require_epsilon_max_ratio", type=float, default=0.1)
    parser.add_argument("--change_job_epsilon_max_times", type=float, default=1.0)

    parser.add_argument("--base_capacity", type=float, default=10.0)
    parser.add_argument("--budget_capacity_ratio", type=float, default=1)
    parser.add_argument("--change_datablock_epsilon_max_times", type=float, default=1.0)
    
    parser.add_argument("--scheduler_update_sleep_time", type=float, default=0.0)
    parser.add_argument("--cal_significance_sleep_time", type=float, default=0.0)
    parser.add_argument("--placement_sleep_time", type=float, default=1.0)
    parser.add_argument("--sched_best_serve_sleep_time", type=float, default=1.0)

    parser.add_argument("--waiting_time", type=int, default=2)
    parser.add_argument("--update_timeout", type=int, default=500)
    parser.add_argument("--without_start_load_job", action="store_true")
    parser.add_argument("--without_start_load_history_job", action="store_true")
    parser.add_argument("--without_start_load_dataset", action="store_true")
    parser.add_argument("--without_finished_clear_all", action="store_true")
    parser.add_argument("--without_stop_all", action="store_true")

    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])

    parser.add_argument("--assignment_policy", type=str, default="HISPolicy")
    parser.add_argument("--significance_policy", type=str, default="HISOTDDPolicy")

    parser.add_argument('--pbg_comparison_cost_epsilons', type=float, default=0.0)
    parser.add_argument('--pbg_comparison_z_thresholds', type=float, default=0.9)
    parser.add_argument('--pbg_Ls', type=float, default=0.01)
    parser.add_argument('--pbg_Us', type=float, default=0.1) # 0.1
    parser.add_argument('--pbg_gittas', type=float, default=0.9)

    parser.add_argument('--his_betas', type=float, default=0.0)
    parser.add_argument('--his_batch_size_for_one_epochs', type=int, default=25)
    # parser.add_argument('--his_comparison_z_threshold_list', type=float, default=0.7)

    parser.add_argument('--dpf_his_betas', type=float, default=0.01)
    parser.add_argument('--dpf_his_waiting_queue_capacitys', type=int, default=10)

    parser.add_argument("--worker_ips", type=str, nargs="+", default=["172.18.162.2", "172.18.162.3", "172.18.162.4", "172.18.162.5", "172.18.162.6"])
    parser.add_argument("--worker_ports", type=int, nargs="+", default=[16202, 16203, 16204, 16205, 16206])
    parser.add_argument("--worker_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    
    parser.add_argument("--sched_ip", type=str, default="172.18.162.6")
    parser.add_argument("--sched_port", type=int, default=16306)

    parser.add_argument("--dispatcher_ip", type=str, default="172.18.162.6")
    parser.add_argument("--dispatcher_port", type=int, default=16406)
    
    args = parser.parse_args()
    return args

class Dispatcher(object):
    def __init__(self):
        self.all_finished = True
    
    def restart_dispatcher(self, jobs_list, history_jobs_list, datasets_map, current_test_all_dir, simulation, simulation_index, restart_trace):
        self.current_test_all_dir = current_test_all_dir
        all_logger_path = '{}/{}'.format(RESULT_PATH, self.current_test_all_dir)
        if restart_trace:
            dispatcher_logger_path = '{}/DL_dispatcher_restart_{}.log'.format(all_logger_path, simulation_index)
        else:
            dispatcher_logger_path = '{}/DL_dispatcher_{}.log'.format(all_logger_path, simulation_index)
        self.dispatcher_logger = get_logger(dispatcher_logger_path, dispatcher_logger_path, enable_multiprocess=True)
        self.dispatcher_logger.info("***************** current_test_all_dir: {} *****************".format(self.current_test_all_dir))

        jobs_list = sorted(jobs_list, key=lambda r: r["time"])
        jobs_id_list = ["job_{}".format(x) for x in range(len(jobs_list))]
        jobs_detail = list(map(lambda x: [x[0], x[1]], zip(jobs_id_list, jobs_list)))

        history_jobs_detail = sorted(history_jobs_list, key=lambda r: r["time"])
        history_jobs_detail = {"history_job_{}".format(i): history_jobs_list[i] for i in range(len(history_jobs_list))}

        self.jobs_detail = jobs_detail
        self.history_jobs_detail = history_jobs_detail

        self.finished_labels = {job_id:False for job_id, _ in self.jobs_detail}
        self.dispatch_jobs_count = 0

        self.datasets_map = datasets_map
        count = 0
        for sub_map in datasets_map.values():
            count += len(sub_map)
        self.all_datasets_count = count
        self.dispatch_datasets_count = 0

        self.all_start_time = time.time()
        self.current_time = 0
        self.all_finished = False

    def end_and_report_by_sched(self, all_result_map):
        current_success_num = all_result_map["current_success_num"]
        current_failed_num = all_result_map["current_failed_num"]
        current_no_submit_num = all_result_map["current_no_submit_num"]
        current_no_sche_num = all_result_map["current_no_sche_num"]
        current_done_sig_num = all_result_map["current_done_sig_num"]
        current_done_sche_num = all_result_map["current_done_sche_num"]
        current_running_num = all_result_map["current_running_num"]
        current_recoming_num = all_result_map["current_recoming_num"]
        all_train_loss = all_result_map["all_train_loss"]
        all_train_accuracy = all_result_map["all_train_accuracy"]
        all_test_loss = all_result_map["all_test_loss"]
        all_test_accuracy = all_result_map["all_test_accuracy"]
        all_final_significance = all_result_map["all_final_significance"]
        all_target_datablock_num = all_result_map["all_target_datablock_num"]
        all_success_datablock_num = all_result_map["all_success_datablock_num"]
        all_failed_datablock_num = all_result_map["all_failed_datablock_num"]

        self.dispatcher_logger.debug("current_success_num: {}; current_failed_num: {}; current_no_submit_num: {}; current_no_sche_num: {};".format(
            current_success_num, current_failed_num, current_no_submit_num, current_no_sche_num
        ))
        self.dispatcher_logger.debug("current_done_sig_num: {}; current_done_sche_num: {}; current_running_num: {}; current_recoming_num: {};".format(
            current_done_sig_num, current_done_sche_num, current_running_num, current_recoming_num
        ))
        job_sequence_all_num = len(self.jobs_detail)
        self.dispatcher_logger.debug("all test jobs num: {}".format(job_sequence_all_num))
        self.dispatcher_logger.debug("all_train_loss: {}".format(all_train_loss / job_sequence_all_num))
        self.dispatcher_logger.debug("all_train_accuracy: {}".format(all_train_accuracy / job_sequence_all_num))
        self.dispatcher_logger.debug("all_test_loss: {}".format(all_test_loss / job_sequence_all_num))
        self.dispatcher_logger.debug("all_test_accuracy: {}".format(all_test_accuracy / job_sequence_all_num))
        self.dispatcher_logger.debug("all_final_significance: {}".format(all_final_significance / job_sequence_all_num))
        
        self.dispatcher_logger.debug("all_target_datablock_num: {}".format(all_target_datablock_num))
        self.dispatcher_logger.debug("all_success_datablock_num: {}".format(all_success_datablock_num))
        self.dispatcher_logger.debug("all_failed_datablock_num: {}".format(all_failed_datablock_num))

        if current_success_num > 0:
            self.dispatcher_logger.debug("success_train_loss: {}".format(all_train_loss / current_success_num))
            self.dispatcher_logger.debug("success_train_accuracy: {}".format(all_train_accuracy / current_success_num))
            self.dispatcher_logger.debug("success_test_loss: {}".format(all_test_loss / current_success_num))
            self.dispatcher_logger.debug("success_test_accuracy: {}".format(all_test_accuracy / current_success_num))
            self.dispatcher_logger.debug("success_final_significance: {}".format(all_final_significance / current_success_num))
        
        self.all_finished = True
        

    def dispatch_jobs(self, all_decision_num, sched_ip, sched_port, update_timeout):
        def thread_func_timely_dispatch_job(sched_ip, sched_port, update_timeout):
            while not self.all_finished:
                count = self.dispatch_jobs_count
                dispatch_jobs_detail = {}
                for index in range(len(self.jobs_detail)):
                    job_id, info = self.jobs_detail[index]
                    
                    need_submit_time = info["time"]
                    has_submited_flag = info["submited"]
                    if not has_submited_flag and need_submit_time <= self.current_time:
                        self.dispatcher_logger.info("[add job start job_id: {}] need_submit_time: {}; self.current_time: {}".format(job_id, need_submit_time, self.current_time))
                        self.jobs_detail[index][1]["submited"] = True
                        count += 1
                        dispatch_jobs_detail[job_id] = info
                if count > self.dispatch_jobs_count:
                    self.dispatch_jobs_count = count
                    client = self.get_zerorpc_client(sched_ip, sched_port, timeout=update_timeout)
                    client.update_jobs(dispatch_jobs_detail) # 提交上去后, 任务即进入NO_SCHED状态, 之后就是调度器自身会启动一个不断循环的获取计算Siginificane策略和调度策略
                if self.dispatch_datasets_count == len(self.jobs_detail):
                    client = self.get_zerorpc_client(sched_ip, sched_port, timeout=update_timeout)
                    client.set_final_job_flag(True)
                    self.dispatcher_logger.info("Finished Job Dispatch!")
                    break
                time.sleep(1)
            self.dispatcher_logger.info("Thread [thread_func_timely_dispatch_job] finished!")
        # 在最开始一定要将真实的历史结果传过去
        client = self.get_zerorpc_client(sched_ip, sched_port)
        client.init_jobs_all_sequence_num(all_decision_num)
        p = threading.Thread(target=thread_func_timely_dispatch_job, args=(sched_ip, sched_port, update_timeout), daemon=True)
        p.start()
        return p

    def dispatch_history_jobs(self, sched_ip, sched_port, update_timeout):
        def thread_func_once_dispatch_his_job(sched_ip, sched_port, update_timeout):
            client = self.get_zerorpc_client(sched_ip, sched_port, timeout=update_timeout)
            client.update_history_jobs(self.history_jobs_detail)
            self.dispatcher_logger.info("Thread [thread_func_once_dispatch_his_job] finished!")
        p = threading.Thread(target=thread_func_once_dispatch_his_job, args=(sched_ip, sched_port, update_timeout), daemon=True)
        p.start()
        return p

    def finished_job_callback(self, job_id):
        current_time = time.time()
        self.dispatcher_logger.info("[finished job end job_id: {}] current_time: {};".format(job_id, current_time))
        self.finished_labels[job_id] = True   

    def failed_job_callback(self, job_id):
        current_time = time.time()
        self.dispatcher_logger.info("[failed job end job_id: {}] current_time: {}".format(job_id, current_time))
        self.finished_labels[job_id] = True

    def sched_update_dataset(self, sched_ip, sched_port, update_timeout):
        def thread_func_timely_dispatch_dataset(sched_ip, sched_port, update_timeout):
            while not self.all_finished:
                count = self.dispatch_datasets_count
                subtrain_datasetidentifier_info = {}
                for dataset_name in self.datasets_map:
                    for sub_train_dataset_identifier in self.datasets_map[dataset_name]:
                        need_submit_time = self.datasets_map[dataset_name][sub_train_dataset_identifier]["time"]
                        epsilon_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["epsilon_capacity"]
                        delta_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["delta_capacity"]
                        has_submited_flag = self.datasets_map[dataset_name][sub_train_dataset_identifier]["submited"]
                        if not has_submited_flag and need_submit_time <= self.current_time:
                            self.dispatcher_logger.info("[add dataset start dataset_name: {}; sub_train_dataset_identifier: {}]  need_submit_time: {}; self.current_time: {}".format(dataset_name, sub_train_dataset_identifier, need_submit_time, self.current_time))
                            self.datasets_map[dataset_name][sub_train_dataset_identifier]["submited"] = True
                            count += 1
                            if dataset_name not in subtrain_datasetidentifier_info:
                                subtrain_datasetidentifier_info[dataset_name] = {}
                            subtrain_datasetidentifier_info[dataset_name][sub_train_dataset_identifier] = {
                                "epsilon_capacity": epsilon_capacity,
                                "delta_capacity": delta_capacity,
                            }
                if count > self.dispatch_datasets_count:
                    self.dispatch_datasets_count = count
                    client = self.get_zerorpc_client(sched_ip, sched_port, timeout=update_timeout)
                    client.update_dataset(subtrain_datasetidentifier_info)
                if self.dispatch_datasets_count == self.all_datasets_count:
                    self.dispatcher_logger.info("Finished Dataset Dispatch!")
                    break
                time.sleep(1)
            self.dispatcher_logger.info("Thread [thread_func_timely_dispatch_dataset] finished!")
        p = threading.Thread(target=thread_func_timely_dispatch_dataset, args=(sched_ip, sched_port, update_timeout), daemon=True)
        p.start()
        return p

    
    def sched_update_current_time(self):
        def thread_func_timely_update_time():
            while not self.all_finished:
                self.current_time = time.time() - self.all_start_time
                time.sleep(1)
            self.dispatcher_logger.info("Thread [thread_func_timely_update_time] finished!")
        p = threading.Thread(target=thread_func_timely_update_time, daemon=True)
        p.start()
        return p

    def sched_simulation_start(self, sched_ip, sched_port):
        subtrain_datasetidentifier_info = {}
        for dataset_name in self.datasets_map:
            for sub_train_dataset_identifier in self.datasets_map[dataset_name]:
                need_submit_time = self.datasets_map[dataset_name][sub_train_dataset_identifier]["time"]
                epsilon_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["epsilon_capacity"]
                delta_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["delta_capacity"]
                if dataset_name not in subtrain_datasetidentifier_info:
                    subtrain_datasetidentifier_info[dataset_name] = {}
                subtrain_datasetidentifier_info[dataset_name][sub_train_dataset_identifier] = {
                    "time": need_submit_time,
                    "epsilon_capacity": epsilon_capacity,
                    "delta_capacity": delta_capacity,
                }
        temp_history_job_details = {}
        for history_job_id in self.history_jobs_detail:
            info = self.history_jobs_detail[history_job_id]
            del_info = copy.deepcopy(info)
            temp_history_job_details[history_job_id] = del_info

        temp_submit_job_details = []
        for index in range(len(self.jobs_detail)):
            job_id, info = self.jobs_detail[index]
            del_info = copy.deepcopy(info)
            del del_info["submited"] 
            temp_submit_job_details.append([job_id, del_info])
        subtrain_datasetidentifier_info = convert_types(subtrain_datasetidentifier_info)
        temp_history_job_details = convert_types(temp_history_job_details)
        temp_submit_job_details = convert_types(temp_submit_job_details)
        all_decision_num = len(temp_submit_job_details)
        client = self.get_zerorpc_client(sched_ip, sched_port)
        client.init_jobs_all_sequence_num(all_decision_num)
        client.sched_simulation_start(subtrain_datasetidentifier_info, temp_history_job_details, temp_submit_job_details)
    
    def get_zerorpc_client(self, ip, port, timeout=30):
        tcp_ip_port = "tcp://{}:{}".format(ip, port)
        client = zerorpc.Client(timeout=timeout)
        client.connect(tcp_ip_port)
        return client

    def sched_clear_all(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.clear_all()

    def stop_all(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.stop_all()

    def sched_dispatch_start(self, ip, port, cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_time, sched_best_serve_sleep_time):
        client = self.get_zerorpc_client(ip, port)
        # client.sched_dispatch_testbed_start(cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_time)
        client.cal_significance_dispatch_start(cal_significance_sleep_time)
        client.sched_dispatch_start(scheduler_update_sleep_time)
        client.placement_dispatch_start(placement_sleep_time)
        # client.sched_best_serve_start(sched_best_serve_sleep_time)
    
    def sched_end(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.sched_end()

    def sched_report_status(self, ip, port, location):
        client = self.get_zerorpc_client(ip, port)
        client.report_status(location)

    def sched_update_gpu_status_start(self, ip, port, init_workerip_2_ports, init_gpuidentifiers, current_test_all_dir, simulation_index):
        client = self.get_zerorpc_client(ip, port)
        client.sched_update_gpu_status_start(init_workerip_2_ports, init_gpuidentifiers, current_test_all_dir, simulation_index)

    def sched_init_sched_register(self, ip, port, args, seed, assignment_policy, significance_policy, all_decision_num, simulation, simulation_index):
        client = self.get_zerorpc_client(ip, port)
        client.restart_sched()
        client.initialize_seeds(seed)
        client.initialize_simulation_flag(simulation, simulation_index)
        client.initialize_logging_path(self.current_test_all_dir, simulation_index)
        if assignment_policy == "PBGPolicy":
            comparison_cost_epsilon_list = args.pbg_comparison_cost_epsilons
            comparison_z_threshold_list = args.pbg_comparison_z_thresholds
            L_list = args.pbg_Ls
            U_list = args.pbg_Us
            assignment_args = (comparison_cost_epsilon_list, comparison_z_threshold_list, L_list, U_list)
        elif assignment_policy == "PBGMixPolicy":
            comparison_cost_epsilon_list = args.pbg_comparison_cost_epsilons
            comparison_z_threshold_list = args.pbg_comparison_z_thresholds
            L_list = args.pbg_Ls
            U_list = args.pbg_Us
            gitta_list = args.pbg_gittas
            assignment_args = (comparison_cost_epsilon_list, comparison_z_threshold_list, L_list, U_list, gitta_list)
        elif assignment_policy == "HISPolicy" or assignment_policy == "HISwithCPolicy" or assignment_policy == "HISwithOrderRemainVersionPolicy" or assignment_policy == "HISwithOrderProVersionPolicy":
            beta_list = args.his_betas
            assignment_args = (beta_list, all_decision_num)
        elif assignment_policy == "IterativeHISPolicy" or assignment_policy == "IterativeHISwithOrderProVersionPolicy" or assignment_policy == "IterativeHISwithOrderRemainVersionPolicy":
            beta_list = args.his_betas
            batch_size_for_one_epoch_list = args.his_batch_size_for_one_epochs
            assignment_args = (beta_list, batch_size_for_one_epoch_list, all_decision_num)
        elif assignment_policy == "DPFHISPolicy":
            beta_list = args.dpf_his_betas
            waiting_queue_capacity_list = args.dpf_his_waiting_queue_capacitys
            assignment_args = (beta_list, waiting_queue_capacity_list, all_decision_num)
        elif assignment_policy == "OfflinePolicy":
            assignment_args = all_decision_num
        else:
            assignment_args = None
        client.sched_update_assignment_policy(assignment_policy, assignment_args)
        client.sched_update_significance_policy(significance_policy)
        self.dispatcher_logger.info("sched_init_sched_register finished!")
        args_message = '\n'.join([f'{k}: {v}' for k, v in vars(args).items()])
        self.dispatcher_logger.info("===== args_message =====")
        self.dispatcher_logger.info(args_message)

    def sched_init_history_policy(self, ip, port, history_jobs_map):
        client = self.get_zerorpc_client(ip, port)
        client.update_history_jobs(history_jobs_map)

    def final_operate_data(self, current_test_all_dir):
        trace_save_path = "{}/{}".format(RESULT_PATH, current_test_all_dir)
        final_used_num, success_num_arr, failed_num_arr, all_final_significance_arr, success_final_significance_arr, \
            success_datablock_num_arr, failed_datablock_num_arr, target_datablock_num_arr = read_DL_dispatcher_result_func(trace_save_path)
        
        # 新建一个全新的log进行保存
        all_result_path = "{}/all_result.log".format(trace_save_path)
        with open(all_result_path, "w+") as f:
            print("final_used_num: {}".format(final_used_num))
            print("final_used_num: {}".format(final_used_num), file=f)
            print("[success_info] {}({}-{}) === success_num_mean: {} ; success_num_min: {} ; success_num_max: {}".format(
                np.mean(success_num_arr), min(success_num_arr), max(success_num_arr), np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
            ))
            print("[success_info] {}({}-{}) === success_num_mean: {} ; success_num_min: {} ; success_num_max: {}".format(
                np.mean(success_num_arr), min(success_num_arr), max(success_num_arr), np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
            ), file=f)
            print("[failed_info] {}({}-{}) === failed_num_mean: {} ; failed_num_min: {} ; failed_num_max: {}".format(
                np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr), np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
            ))
            print("[failed_info] {}({}-{}) === failed_num_mean: {} ; failed_num_min: {} ; failed_num_max: {}".format(
                np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr), np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
            ), file=f)
            print("[allsig_info] {}({}-{}) === all_final_significance_mean: {} ; all_final_significance_min: {} ; all_final_significance_max: {}".format(
                np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr), np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
            ))
            print("[allsig_info] {}({}-{}) === all_final_significance_mean: {} ; all_final_significance_min: {} ; all_final_significance_max: {}".format(
                np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr), np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
            ), file=f)
            print("[successsig_info] {}({}-{}) === success_final_significance_mean: {} ; success_final_significance_min: {} ; success_final_significance_max: {}".format(
                np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr), np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr)
            ))
            print("[successsig_info] {}({}-{}) === success_final_significance_mean: {} ; success_final_significance_min: {} ; success_final_significance_max: {}".format(
                np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr), np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr)
            ), file=f)

            print("[successblock_info] {}({}-{}) === success_datablock_num_mean: {} ; success_datablock_num_min: {} ; success_datablock_num_max: {}".format(
                np.mean(success_datablock_num_arr), min(success_datablock_num_arr), max(success_datablock_num_arr), np.mean(success_datablock_num_arr), min(success_datablock_num_arr), max(success_datablock_num_arr)
            ))
            print("[successblock_info] {}({}-{}) === success_datablock_num_mean: {} ; success_datablock_num_min: {} ; success_datablock_num_max: {}".format(
                np.mean(success_datablock_num_arr), min(success_datablock_num_arr), max(success_datablock_num_arr), np.mean(success_datablock_num_arr), min(success_datablock_num_arr), max(success_datablock_num_arr)
            ), file=f)
            print("[failedblock_info] {}({}-{}) === failed_datablock_num_mean: {} ; failed_datablock_num_min: {} ; failed_datablock_num_max: {}".format(
                np.mean(failed_datablock_num_arr), min(failed_datablock_num_arr), max(failed_datablock_num_arr), np.mean(failed_datablock_num_arr), min(failed_datablock_num_arr), max(failed_datablock_num_arr)
            ))
            print("[failedblock_info] {}({}-{}) === failed_datablock_num_mean: {} ; failed_datablock_num_min: {} ; failed_datablock_num_max: {}".format(
                np.mean(failed_datablock_num_arr), min(failed_datablock_num_arr), max(failed_datablock_num_arr), np.mean(failed_datablock_num_arr), min(failed_datablock_num_arr), max(failed_datablock_num_arr)
            ), file=f)
            print("[targetblock_info] {}({}-{}) === target_datablock_num_mean: {} ; target_datablock_num_min: {} ; target_datablock_num_max: {}".format(
                np.mean(target_datablock_num_arr), min(target_datablock_num_arr), max(target_datablock_num_arr), np.mean(target_datablock_num_arr), min(target_datablock_num_arr), max(target_datablock_num_arr)
            ))
            print("[targetblock_info] {}({}-{}) === target_datablock_num_mean: {} ; target_datablock_num_min: {} ; target_datablock_num_max: {}".format(
                np.mean(target_datablock_num_arr), min(target_datablock_num_arr), max(target_datablock_num_arr), np.mean(target_datablock_num_arr), min(target_datablock_num_arr), max(target_datablock_num_arr)
            ), file=f)

def scheduler_listener_func(dispatcher_server_item, port):
    def dispatcher_func_timely(dispatcher_server_item, port):
        dispatcher_server = zerorpc.Server(dispatcher_server_item)
        ip_port = "tcp://0.0.0.0:{}".format(port)
        dispatcher_server.bind(ip_port)
        print("DL_server running in {}".format(ip_port))
        dispatcher_server.run()
        print("Thread [dispatcher_func_timely] finished!")
    p = threading.Thread(target=dispatcher_func_timely, args=(dispatcher_server_item, port), daemon=True)
    p.start()
    return p

def exit_gracefully(server):
    print("stopping server")
    server.stop()
    print("closing server")
    server.close()



def testbed_experiment_start(args, sched_ip, sched_port,
                            dispatcher_ip, dispatcher_port,
                            worker_ips, worker_ports, init_gpuidentifiers, init_workerip_2_ports,
                            global_sleep_time, update_timeout, scheduler_update_sleep_time, 
                            cal_significance_sleep_time, sched_best_serve_sleep_time, 
                            placement_sleep_time, waiting_time,
                            dataset_reconstruct_path, test_jobtrace_reconstruct_path, history_jobtrace_reconstruct_path,
                            budget_capacity_ratio, base_capacity, change_datablock_epsilon_max_times,
                            job_dataset_trace_save_path, current_test_all_dir, restart_trace,
                            all_decision_num, all_history_num, time_interval, need_change_interval,
                            datablock_require_epsilon_max_ratio, change_job_epsilon_max_times):
    assert args.simulation_time == 1 and len(args.seeds) == 1
    simulation_flag = False
    block_global_epsilon = base_capacity * budget_capacity_ratio
    job_epsilon_ub = block_global_epsilon * datablock_require_epsilon_max_ratio
    job_epsilon_lb = 0.02
    datasets_list = generate_dataset(
        dataset_names=["EMNIST"], 
        fix_epsilon=base_capacity * budget_capacity_ratio, 
        fix_delta=1e-5, 
        change_datablock_epsilon_max_times=change_datablock_epsilon_max_times,
        fix_time=0, 
        num=6, 
        dataset_reconstruct_path=dataset_reconstruct_path, 
        save_path=current_test_all_dir
    )
    jobs_list = generate_jobs(
        all_num=all_decision_num,
        per_epoch_EPSILONs=[job_epsilon_lb, job_epsilon_ub],
        datablock_require_epsilon_max_ratio=datablock_require_epsilon_max_ratio,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        time_interval=time_interval, 
        need_change_interval=need_change_interval, 
        is_history=False, 
        dispatcher_ip=dispatcher_ip, 
        dispatcher_port=dispatcher_port, 
        jobtrace_reconstruct_path=test_jobtrace_reconstruct_path, 
        save_path=current_test_all_dir
    )
    history_jobs_list = generate_jobs(
        all_num=all_history_num, 
        per_epoch_EPSILONs=[job_epsilon_lb, job_epsilon_ub],
        datablock_require_epsilon_max_ratio=datablock_require_epsilon_max_ratio,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        time_interval=time_interval, 
        need_change_interval=need_change_interval, 
        is_history=True, 
        dispatcher_ip=dispatcher_ip, 
        dispatcher_port=dispatcher_port, 
        jobtrace_reconstruct_path=history_jobtrace_reconstruct_path, 
        save_path=current_test_all_dir
    )
    all_decision_num = len(jobs_list)

    processes = []
    try:
        dispatcher = Dispatcher()
        dispatcher.restart_dispatcher(
            jobs_list, 
            history_jobs_list, 
            datasets_list, 
            current_test_all_dir,
            simulation=False,
            simulation_index=0,
            restart_trace=restart_trace
        )
        
        remote_server_p = scheduler_listener_func(dispatcher, dispatcher_port)
        processes.append(remote_server_p)

        dispatcher.sched_init_sched_register(
            sched_ip, sched_port, 
            args, args.seeds[0], 
            args.assignment_policy, args.significance_policy, 
            all_decision_num, 
            simulation=False, simulation_index=0)
        dispatcher.sched_update_gpu_status_start(
            sched_ip, sched_port, 
            init_workerip_2_ports, init_gpuidentifiers, 
            current_test_all_dir, simulation_index=0
        )
        if not args.without_start_load_job:
            dataset_p = dispatcher.sched_update_dataset(sched_ip, sched_port, update_timeout)
            processes.append(dataset_p)
        
        time.sleep(waiting_time)
        time_p = dispatcher.sched_update_current_time()
        processes.append(time_p)

        if not args.without_start_load_history_job:
            history_job_p = dispatcher.dispatch_history_jobs(sched_ip, sched_port, update_timeout)
            processes.append(history_job_p)
        if not args.without_start_load_job:
            job_p = dispatcher.dispatch_jobs(all_decision_num, sched_ip, sched_port, update_timeout)
            processes.append(job_p)

        print("Waiting for load datasets and jobs {} s".format(waiting_time))
        time.sleep(waiting_time)
        dispatcher.sched_dispatch_start(sched_ip, sched_port, cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_time, sched_best_serve_sleep_time)
        
        # 主线程的最后一个操作!
        all_finished_label = reduce(lambda a, b: a and b, dispatcher.finished_labels.values())
        while not all_finished_label:
            time.sleep(global_sleep_time)
            all_finished_label = reduce(lambda a, b: a and b, dispatcher.finished_labels.values())
        dispatcher.sched_report_status(sched_ip, sched_port, "all stop")
        print("logically all stoped!")
        dispatcher.sched_end(sched_ip, sched_port)
        if not args.without_finished_clear_all:
            dispatcher.sched_clear_all(sched_ip, sched_port)
        print("Stop workers and scheduler")
        time.sleep(waiting_time)
        
        if not args.without_stop_all:
            dispatcher.stop_all(sched_ip, sched_port)
        print("Waiting for stop threads {} s".format(waiting_time))
        time.sleep(waiting_time)
    except Exception as e:
        print("[xlc] Exception: ", e)

def simulation_experiment_start(args, sched_ip, sched_port,
                            dispatcher_ip, dispatcher_port,
                            worker_ips, worker_ports, init_gpuidentifiers, init_workerip_2_ports,
                            dataset_reconstruct_path, test_jobtrace_reconstruct_path, history_jobtrace_reconstruct_path,
                            budget_capacity_ratio, base_capacity, change_datablock_epsilon_max_times, 
                            job_dataset_trace_save_path, current_test_all_dir, restart_trace,
                            all_decision_num, all_history_num, need_change_interval,
                            simulation_time, simulation_time_speed_up, simulation_all_datablock_num, simulation_offline_datablock_num, 
                            datablock_require_epsilon_max_ratio, change_job_epsilon_max_times
                            ):
    min_epsilon_capacity = base_capacity * budget_capacity_ratio
    datasets_list = generate_alibaba_dataset(
        num=simulation_all_datablock_num,
        offline_num=simulation_offline_datablock_num,
        time_speed_up=simulation_time_speed_up,
        dataset_names=["EMNIST"],
        fix_epsilon=min_epsilon_capacity,
        fix_delta=1e-5,
        change_datablock_epsilon_max_times=change_datablock_epsilon_max_times,
        dataset_reconstruct_path=dataset_reconstruct_path, 
        save_path=job_dataset_trace_save_path
    )
    jobs_list = generate_alibaba_jobs(
        all_num=all_decision_num,
        time_speed_up=simulation_time_speed_up,
        need_change_interval=need_change_interval,
        is_history=False,
        datablock_require_epsilon_max_ratio=datablock_require_epsilon_max_ratio,
        min_epsilon_capacity=min_epsilon_capacity,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        dispatcher_ip=dispatcher_ip,
        dispatcher_port=dispatcher_port,
        jobtrace_reconstruct_path=test_jobtrace_reconstruct_path,
        save_path=job_dataset_trace_save_path
    )
    history_jobs_list = generate_alibaba_jobs(
        all_num=all_history_num,
        time_speed_up=simulation_time_speed_up,
        need_change_interval=need_change_interval,
        is_history=True,
        datablock_require_epsilon_max_ratio=datablock_require_epsilon_max_ratio,
        min_epsilon_capacity=min_epsilon_capacity,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        dispatcher_ip=dispatcher_ip,
        dispatcher_port=dispatcher_port,
        jobtrace_reconstruct_path=history_jobtrace_reconstruct_path,
        save_path=job_dataset_trace_save_path
    )
    all_decision_num = len(jobs_list)
    processes = []
    dispatcher = Dispatcher()
    remote_server_p = scheduler_listener_func(dispatcher, dispatcher_port)
    processes.append(remote_server_p)
    for simulation_index in range(simulation_time):
        print("start simulation_index: {}".format(simulation_index))
        try:
            dispatcher.restart_dispatcher(
                jobs_list, 
                history_jobs_list, 
                datasets_list, 
                current_test_all_dir, 
                simulation=True, 
                simulation_index=simulation_index,
                restart_trace=restart_trace)
            
            dispatcher.sched_init_sched_register(
                sched_ip, sched_port, 
                args, args.seeds[simulation_index], 
                args.assignment_policy, args.significance_policy, 
                all_decision_num, 
                simulation=True,
                simulation_index=simulation_index
            )
            dispatcher.sched_update_gpu_status_start(
                sched_ip, sched_port, 
                init_workerip_2_ports, init_gpuidentifiers, 
                current_test_all_dir, simulation_index
            )

            # 合并成统一的事件队列, 需要等待所有的
            dispatcher.sched_simulation_start(sched_ip, sched_port) # 同样直接启动一个线程, 完成一大堆队列操作即可

            # 主线程的最后一个操作!
            while not dispatcher.all_finished:
                time.sleep(global_sleep_time)
            dispatcher.sched_report_status(sched_ip, sched_port, "all stop")
            print("logically all stoped!")
            dispatcher.all_finished = True
            dispatcher.sched_end(sched_ip, sched_port)
            if not args.without_finished_clear_all:
                dispatcher.sched_clear_all(sched_ip, sched_port)
            print("Stop workers and scheduler")
            time.sleep(waiting_time)
        except Exception as e:
            print("[xlc] Exception: ", e)
        print("end simulation_index: {}".format(simulation_index))
    if not args.without_stop_all:
        dispatcher.stop_all(sched_ip, sched_port)
    dispatcher.final_operate_data(current_test_all_dir) # 执行数据的处理和绘图操作
    print("Waiting for stop threads {} s".format(waiting_time))
    time.sleep(waiting_time)

if __name__ == "__main__":
    args = get_df_config()

    sched_ip = args.sched_ip
    sched_port = args.sched_port
    dispatcher_ip = args.dispatcher_ip
    dispatcher_port = args.dispatcher_port
    worker_ips = args.worker_ips
    worker_ports = args.worker_ports
    worker_indexes = args.worker_indexes
    assert len(worker_ips) == len(worker_ports)
    init_gpuidentifiers = []
    init_workerip_2_ports = {}
    for worker_ip, worker_port in zip(worker_ips, worker_ports):
        init_workerip_2_ports[worker_ip] = worker_port
        for i in worker_indexes:
            temp_identifier = worker_ip + "-{}".format(i)
            init_gpuidentifiers.append(temp_identifier)

    dataset_reconstruct_path = args.dataset_reconstruct_path
    test_jobtrace_reconstruct_path = args.test_jobtrace_reconstruct_path
    history_jobtrace_reconstruct_path = args.history_jobtrace_reconstruct_path
    save_to_origin_trace_path = args.save_to_origin_trace_path

    budget_capacity_ratio = args.budget_capacity_ratio
    base_capacity = args.base_capacity
    change_datablock_epsilon_max_times = args.change_datablock_epsilon_max_times

    if len(save_to_origin_trace_path) > 0:
        current_test_all_dir = save_to_origin_trace_path
        restart_trace = True
    else:
        logging_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
        schedule_type = 'simulation' if args.simulation_flag else 'testbed'
        current_test_all_dir = 'schedule-review-%s-%s' % (schedule_type, logging_time)
        restart_trace = False
    job_dataset_trace_save_path = current_test_all_dir if args.need_save_jobtrace_flag else ""

    all_decision_num = args.all_decision_num
    all_history_num = args.all_history_num
    time_interval = args.time_interval
    need_change_interval = args.need_change_interval

    simulation_time = args.simulation_time
    simulation_time_speed_up = args.simulation_time_speed_up
    simulation_all_datablock_num = args.simulation_all_datablock_num
    simulation_offline_datablock_num = args.simulation_offline_datablock_num

    datablock_require_epsilon_max_ratio = args.datablock_require_epsilon_max_ratio
    change_job_epsilon_max_times = args.change_job_epsilon_max_times

    global_sleep_time = args.global_sleep_time 
    update_timeout = args.update_timeout 
    scheduler_update_sleep_time = args.scheduler_update_sleep_time
    cal_significance_sleep_time = args.cal_significance_sleep_time 
    sched_best_serve_sleep_time = args.sched_best_serve_sleep_time
    placement_sleep_time = args.placement_sleep_time 
    waiting_time = args.waiting_time

    if args.simulation_flag:
        simulation_experiment_start(args, sched_ip, sched_port,
                            dispatcher_ip, dispatcher_port,
                            worker_ips, worker_ports, init_gpuidentifiers, init_workerip_2_ports,
                            dataset_reconstruct_path, test_jobtrace_reconstruct_path, history_jobtrace_reconstruct_path,
                            budget_capacity_ratio, base_capacity, change_datablock_epsilon_max_times,
                            job_dataset_trace_save_path, current_test_all_dir, restart_trace,
                            all_decision_num, all_history_num, need_change_interval,
                            simulation_time, simulation_time_speed_up, simulation_all_datablock_num, simulation_offline_datablock_num, 
                            datablock_require_epsilon_max_ratio, change_job_epsilon_max_times)
    else:
        testbed_experiment_start(args, sched_ip, sched_port,
                            dispatcher_ip, dispatcher_port,
                            worker_ips, worker_ports, init_gpuidentifiers, init_workerip_2_ports,
                            global_sleep_time, update_timeout, scheduler_update_sleep_time, 
                            cal_significance_sleep_time, sched_best_serve_sleep_time, 
                            placement_sleep_time, waiting_time,
                            dataset_reconstruct_path, test_jobtrace_reconstruct_path, history_jobtrace_reconstruct_path,
                            budget_capacity_ratio, base_capacity, change_datablock_epsilon_max_times,
                            job_dataset_trace_save_path, current_test_all_dir, restart_trace,
                            all_decision_num, all_history_num, time_interval, need_change_interval,
                            datablock_require_epsilon_max_ratio, change_job_epsilon_max_times)    
    
        