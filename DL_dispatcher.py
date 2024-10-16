import zerorpc
import time
from utils.global_variable import RESULT_PATH, RECONSTRUCT_TRACE_PREFIX_PATH
from utils.global_functions import get_types, convert_types, get_zerorpc_client
import threading
import multiprocessing
from functools import reduce
import sys
import argparse
import json
import copy
import os
import numpy as np
import pandas as pd
from queue import PriorityQueue
from utils.logging_tools import get_logger
from utils.generate_tools import generate_alibaba_jobs, generate_alibaba_dataset
from utils.data_operator import final_log_result, log_args_var
import yagmail

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--worker_ips", type=str, nargs="+", default=["172.18.162.2", "172.18.162.3", "172.18.162.4", "172.18.162.5", "172.18.162.6"])
    parser.add_argument("--worker_ports", type=int, nargs="+", default=[16202, 16203, 16204, 16205, 16206])
    parser.add_argument("--simulation_flag", action="store_true")
    parser.add_argument("--simulation_time", type=int, default=1)
    parser.add_argument("--worker_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    
    parser.add_argument("--sched_ip", type=str, default="172.18.162.6")
    parser.add_argument("--sched_port", type=int, default=16306)

    parser.add_argument("--dispatcher_ip", type=str, default="172.18.162.6")
    parser.add_argument("--dispatcher_port", type=int, default=16406)

    parser.add_argument("--dataset_reconstruct_path", type=str, default="")
    parser.add_argument("--test_jobtrace_reconstruct_path", type=str, default="")
    parser.add_argument("--history_jobtrace_reconstruct_path", type=str, default="")

    parser.add_argument("--max_gpu_fuzai", type=int, default=10)
    parser.add_argument("--all_or_nothing_flag", action="store_true")
    parser.add_argument("--enable_waiting_flag", action="store_true")
    parser.add_argument("--need_save_jobtrace_flag", action="store_true")
    parser.add_argument('--inf_job_dispatch_flag', action="store_true")
    parser.add_argument('--need_stop_lower_bound_ratio', type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--waiting_time", type=int, default=2)

    parser.add_argument("--pipeline_sequence_all_num", type=int, default=50)
    parser.add_argument("--all_history_num", type=int, default=50)
    parser.add_argument("--job_arrival_time_speed_up", type=float, default=1.0)
    parser.add_argument("--change_job_epsilon_max_times", type=float, default=1.0) # 这个控制比率(离群值控制)
    parser.add_argument("--job_datablock_epsilon_max_ratio", type=float, default=0.1) # 这个直接从平均增大倍数(平均值控制)
    parser.add_argument("--job_datablock_epsilon_min_ratio", type=float, default=0.01) # 这个直接从平均增大倍数(平均值控制)
    parser.add_argument("--job_require_select_block_min_num", type=float, default=5)
    parser.add_argument("--job_require_select_block_max_num", type=float, default=25)
    parser.add_argument("--config_max_operate_siton_run_num", type=int, default=1)

    parser.add_argument("--all_datablock_num", type=int, default=100)
    parser.add_argument("--offline_datablock_num", type=int, default=100)
    parser.add_argument("--datablock_arrival_time_speed_up", type=float, default=1.0)
    parser.add_argument("--base_capacity", type=float, default=10.0)
    parser.add_argument("--dataset_name", type=str, default="EMNIST")
    parser.add_argument("--dataset_config_name", type=str, default="subtrain_100_split_1.0_dirichlet")

    parser.add_argument("--assignment_policy", type=str, default="HISPolicy")
    parser.add_argument('--his_betas', type=float, default=0.0)
    parser.add_argument('--his_batch_size_for_one_epochs', type=int, default=25)
    parser.add_argument('--his_infinity_flag', action="store_true")
    parser.add_argument('--his_stop_n_growing_flag', action="store_true")
    parser.add_argument('--his_greedy_flag', action="store_true")
    parser.add_argument('--his_greedy_threshold', type=float, default=0.2)
    parser.add_argument('--his_adaptive_cons_generate_flag', type=str, default="None")
    
    parser.add_argument('--pbg_comparison_cost_epsilons', type=float, default=0.0)
    parser.add_argument('--pbg_comparison_z_thresholds', type=float, default=0.9)
    parser.add_argument('--pbg_Ls', type=float, default=0.01)
    parser.add_argument('--pbg_Us', type=float, default=0.1) # 0.1
    parser.add_argument('--pbg_gittas', type=float, default=0.9)

    parser.add_argument("--significance_policy", type=str, default="HISOTDDPolicy")
    parser.add_argument('--temp_sig_metric', type=str, default="Accuracy")
    
    parser.add_argument("--save_to_origin_trace_path", type=str, default="") # 很少

    parser.add_argument("--global_sleep_time", type=int, default=5)
    parser.add_argument("--scheduler_update_sleep_time", type=float, default=1.0)
    parser.add_argument("--cal_significance_sleep_time", type=float, default=1.0)
    parser.add_argument("--placement_sleep_time", type=float, default=2.0)

    
    parser.add_argument("--without_start_load_job", action="store_true")
    parser.add_argument("--without_start_load_history_job", action="store_true")
    parser.add_argument("--without_start_load_dataset", action="store_true")
    parser.add_argument("--without_finished_clear_all", action="store_true")
    parser.add_argument("--without_stop_all", action="store_true")

    args = parser.parse_args()
    return args

class Dispatcher(object):
    def __init__(self, dispatcher_ip, dispatcher_port):
        self.all_finished = True
        self.dispatcher_ip = dispatcher_ip
        self.dispatcher_port = dispatcher_port

    def restart_dispatcher(self, jobs_list, history_jobs_list, datasets_map, 
                            current_test_all_dir, simulation, simulation_index, restart_trace):
        self.current_test_all_dir = current_test_all_dir
        self.simulation = simulation
        self.simulation_index = simulation_index
        all_logger_path = '{}/{}'.format(RESULT_PATH, self.current_test_all_dir)
        if restart_trace:
            self.dispatcher_logger_path = '{}/DL_dispatcher_restart_{}.log'.format(all_logger_path, simulation_index)
        else:
            self.dispatcher_logger_path = '{}/DL_dispatcher_{}.log'.format(all_logger_path, simulation_index)
        self.dispatcher_logger = get_logger(self.dispatcher_logger_path, self.dispatcher_logger_path, enable_multiprocess=True)
        self.dispatcher_logger.info("***************** current_test_all_dir: {} *****************".format(self.current_test_all_dir))

        jobs_list = sorted(jobs_list, key=lambda r: r["time"])
        jobs_id_list = ["job_{}".format(x) for x in range(len(jobs_list))]
        jobs_detail = list(map(lambda x: [x[0], x[1]], zip(jobs_id_list, jobs_list)))

        history_jobs_detail = sorted(history_jobs_list, key=lambda r: r["time"])
        history_jobs_detail = {"history_job_{}".format(i): history_jobs_list[i] for i in range(len(history_jobs_list))}

        self.jobs_detail = jobs_detail
        self.history_jobs_detail = history_jobs_detail

        self.finished_labels = {job_id:False for job_id, _ in self.jobs_detail}
        self.need_judge_finished_label_keys = [job_id for job_id, _ in self.jobs_detail]
        self.dispatch_jobs_count = 0
        

        self.datasets_map = datasets_map
        count = 0
        for sub_map in datasets_map.values():
            count += len(sub_map)
        self.all_datasets_count = count
        self.dispatch_datasets_count = 0

        self.all_start_time = time.time()
        self.current_time = 0

        self.job_result_list = []
        self.all_finished = False

        self.testbed_sched_report_resouce_end = False
        
    def set_testbed_sched_report_resouce_end(self):
        self.testbed_sched_report_resouce_end = True

    def set_need_judge_finished_label_keys(self, temp_need_judge_finished_label_keys):
        self.need_judge_finished_label_keys = temp_need_judge_finished_label_keys

    def testbed_dispatch_jobs(self, inf_job_dispatch_flag, sched_ip, sched_port):
        def thread_func_timely_dispatch_job(sched_ip, sched_port):
            try:
                finished_dispatch_flag = False
                temp_need_judge_finished_label_keys = []
                while not finished_dispatch_flag: 
                    count = self.dispatch_jobs_count
                    dispatch_jobs_detail = {}
                    for index in range(len(self.jobs_detail)):
                        job_id, info = self.jobs_detail[index]
                        
                        need_submit_time = info["time"]
                        has_submited_flag = info["submited"]
                        if (not has_submited_flag) and (need_submit_time <= self.current_time):
                            self.dispatcher_logger.info("[add job start job_id: {}] need_submit_time: {}; self.current_time: {}".format(job_id, need_submit_time, self.current_time))
                            self.jobs_detail[index][1]["submited"] = True
                            count += 1
                            dispatch_jobs_detail[job_id] = info
                            temp_need_judge_finished_label_keys.append(job_id)
                    if count > self.dispatch_jobs_count:
                        self.dispatch_jobs_count = count
                        dispatch_jobs_detail = convert_types(dispatch_jobs_detail)
                        with get_zerorpc_client(sched_ip, sched_port) as client:
                            client.update_jobs(dispatch_jobs_detail) # 提交上去后, 任务即进入NO_SCHED状态, 之后就是调度器自身会启动一个不断循环的获取计算Siginificane策略和调度策略
                    if self.dispatch_jobs_count == len(self.jobs_detail):
                        with get_zerorpc_client(sched_ip, sched_port) as client:
                            client.set_all_job_update_flag()
                        self.dispatcher_logger.info("job max num reach Finished Job Dispatch!")
                        break
                    zerorpc.gevent.sleep(1)
                    if self.all_finished:
                        finished_dispatch_flag = True
                    if inf_job_dispatch_flag and self.testbed_sched_report_resouce_end:
                        finished_dispatch_flag = True
                        self.set_need_judge_finished_label_keys(temp_need_judge_finished_label_keys)
                        self.dispatcher_logger.info(f"sched_report_resouce_end Finished Job Dispatch! change need_judge_finished_label_keys: ({self.need_judge_finished_label_keys}))")
                self.dispatcher_logger.info("Thread [thread_func_timely_dispatch_job] finished!")
            except Exception as e:
                self.dispatcher_logger.error(f"Thread [thread_func_timely_dispatch_job] error => {str(e)}")
                self.dispatcher_logger.exception(e)
        # 在最开始一定要将真实的历史结果传过去
        p = threading.Thread(target=thread_func_timely_dispatch_job, args=(sched_ip, sched_port), daemon=True)
        self.dispatcher_logger.info("Thread [thread_func_timely_dispatch_job] start!")
        p.start()
        return p

    def testbed_dispatch_history_jobs(self, sched_ip, sched_port):
        def thread_func_once_dispatch_his_job(sched_ip, sched_port):
            try:
                history_jobs_detail = convert_types(self.history_jobs_detail)
                with get_zerorpc_client(sched_ip, sched_port) as client:
                    client.update_history_jobs(history_jobs_detail)
                self.dispatcher_logger.info("Thread [thread_func_once_dispatch_his_job] finished!")
            except Exception as e:
                self.dispatcher_logger.error(f"Thread [thread_func_once_dispatch_his_job] error => {str(e)}")
                self.dispatcher_logger.exception(e)
        p = threading.Thread(target=thread_func_once_dispatch_his_job, args=(sched_ip, sched_port), daemon=True)
        p.start()
        self.dispatcher_logger.info("Thread [thread_func_once_dispatch_his_job] start!")
        return p
    
    def operator_job_results_start(self):
        def thread_func_timely_operator_job_result(sleep_time):
            try:
                while not self.all_finished:
                    while len(self.job_result_list) > 0:
                        time, job_id, results, decision_duration, origin_info, success_finished_flag = self.job_result_list.pop(0)
                        self.show_job_results(time, job_id, results, decision_duration, origin_info, success_finished_flag)
                    zerorpc.gevent.sleep(sleep_time)
                self.dispatcher_logger.info(f"Thread [thread_func_timely_operator_job_result] finished")
            except Exception as e:
                self.dispatcher_logger.error(f"Thread [thread_func_timely_operator_job_result] error => {str(e)}")
                self.dispatcher_logger.exception(e)
        p = threading.Thread(target=thread_func_timely_operator_job_result, args=(1, ), daemon=True)
        p.start()
        self.dispatcher_logger.info(f"Thread [thread_func_timely_operator_job_result] start")
        return p

    def show_job_results(self, time, job_id, results, decision_duration, origin_info, success_finished_flag):
        self.dispatcher_logger.info("job [{}] last result: {}".format(job_id, results))
        if len(results) <= 0:
            all_train_loss = 0.0
            all_train_accuracy = 0.0
            all_test_loss = 0.0
            all_test_accuracy = 0.0
            all_final_significance = 0.0
            epsilon_real_all_block = 0.0
            success_datablock_num = 0
            target_datablock_num = origin_info["datablock_select_num"]
        else:
            last_job_res = results[-1]
            if "train_loss" in last_job_res:
                all_train_loss = last_job_res["train_loss"]
            if "train_acc" in last_job_res:
                all_train_accuracy = last_job_res["train_acc"]
            if "test_loss" in last_job_res:
                all_test_loss = last_job_res["test_loss"]
            if "test_acc" in last_job_res:
                all_test_accuracy = last_job_res["test_acc"]
            if "final_significance" in last_job_res:
                all_final_significance = last_job_res["final_significance"]
            
            if "target_datablock_num" in last_job_res:
                target_datablock_num = last_job_res["target_datablock_num"]
            if "success_datablock_num" in last_job_res:
                success_datablock_num = last_job_res["success_datablock_num"]

            if "epsilon_real_all_block" in last_job_res:
                epsilon_real_all_block = last_job_res["epsilon_real_all_block"]
            
            target_datablock_num = origin_info["datablock_select_num"]

        self.dispatcher_logger.info(f"{job_id} train_loss: {all_train_loss}")
        self.dispatcher_logger.info(f"{job_id} train_accuracy: {all_train_accuracy}")
        self.dispatcher_logger.info(f"{job_id} test_loss: {all_test_loss}")
        self.dispatcher_logger.info(f"{job_id} test_accuracy: {all_test_accuracy}")
        self.dispatcher_logger.info(f"{job_id} final_significance: {all_final_significance}")
        self.dispatcher_logger.info(f"{job_id} success_finished_flag: {success_finished_flag}")
        self.dispatcher_logger.info(f"{job_id} target_datablock_num: {target_datablock_num}")
        self.dispatcher_logger.info(f"{job_id} success_datablock_num: {success_datablock_num}")
        self.dispatcher_logger.info(f"{job_id} epsilon_real_all_block: {epsilon_real_all_block}")
        self.dispatcher_logger.info(f"{job_id} decision_duration: {decision_duration}")

        # 在这里要保存每个时刻的记录, 会有资源写入竞争的bug, 最好是写一个队列出来逐步保存!
        xlsx_save_path = "{}/{}/time_temp_log_{}.csv".format(RESULT_PATH, self.current_test_all_dir, self.simulation_index)
        new_data = {
            "time": [time], 
            "job_id": [job_id], 
            
            "train_loss": [all_train_loss],
            "train_acc": [all_train_accuracy],
            "test_loss": [all_test_loss],
            "test_acc": [all_test_accuracy],
            "significance": [all_final_significance],
            "success_flag": [1 if success_finished_flag else 0],
            
            "target_datablock_num": [target_datablock_num],
            "success_datablock_num": [success_datablock_num],

            "epsilon_real_all_block": [epsilon_real_all_block],
            "decision_duration": [decision_duration],
        }
        columns = list(new_data.keys())
        new_df_row = pd.DataFrame(new_data, columns=columns)
        if os.path.exists(xlsx_save_path):
            df = pd.read_csv(xlsx_save_path)
            df = pd.concat([df, new_df_row])
        else:
            df = new_df_row
        df.to_csv(xlsx_save_path, index=False)
        self.dispatcher_logger.debug(f"{job_id} time_log save to csv!")

    def send_job_info_callback(self, job_id, results, decision_duration, origin_info, success_finished_flag):
        if success_finished_flag:
            self.dispatcher_logger.info("[finished job end job_id: {}] current_time: {}".format(job_id, self.current_time))
        else:
            self.dispatcher_logger.info("[failed job end job_id: {}] current_time: {}".format(job_id, self.current_time))
        self.finished_labels[job_id] = True
        self.job_result_list.append((self.current_time, job_id, results, decision_duration, origin_info, success_finished_flag))

    def testbed_sched_update_dataset(self, sched_ip, sched_port):
        def thread_func_timely_dispatch_dataset(sched_ip, sched_port):
            try:
                while not self.all_finished:
                    count = self.dispatch_datasets_count
                    subtrain_datasetidentifier_info = {}
                    for dataset_name in self.datasets_map:
                        for sub_train_dataset_identifier in self.datasets_map[dataset_name]:
                            need_submit_time = self.datasets_map[dataset_name][sub_train_dataset_identifier]["time"]
                            epsilon_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["epsilon_capacity"]
                            delta_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["delta_capacity"]
                            has_submited_flag = self.datasets_map[dataset_name][sub_train_dataset_identifier]["submited"]
                            if (not has_submited_flag) and (need_submit_time <= self.current_time):
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
                        subtrain_datasetidentifier_info = convert_types(subtrain_datasetidentifier_info)
                        with get_zerorpc_client(sched_ip, sched_port) as client:
                            client.update_dataset(subtrain_datasetidentifier_info)
                    if self.dispatch_datasets_count == self.all_datasets_count:
                        self.dispatcher_logger.info("Finished Dataset Dispatch!")
                        with get_zerorpc_client(sched_ip, sched_port) as client:
                            client.set_all_datablock_update_flag()
                        break
                    zerorpc.gevent.sleep(1)
                self.dispatcher_logger.info("Thread [thread_func_timely_dispatch_dataset] finished!")
            except Exception as e:
                self.dispatcher_logger.error(f"Thread [thread_func_timely_dispatch_dataset] error => {str(e)}")
                self.dispatcher_logger.exception(e)
        p = threading.Thread(target=thread_func_timely_dispatch_dataset, args=(sched_ip, sched_port), daemon=True)
        p.start()
        self.dispatcher_logger.info("Thread [thread_func_timely_dispatch_dataset] start!")
        return p

    
    def sched_update_current_time(self):
        def thread_func_timely_update_time():
            while not self.all_finished:
                self.current_time = time.time() - self.all_start_time
                zerorpc.gevent.sleep(1)
            self.dispatcher_logger.info("Thread [thread_func_timely_update_time] finished!")
        p = threading.Thread(target=thread_func_timely_update_time, daemon=True)
        p.start()
        return p

    def sched_simulation_start(self, sched_ip, sched_port):
        try:
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
            with get_zerorpc_client(sched_ip, sched_port) as client:
                client.sched_simulation_start(subtrain_datasetidentifier_info, temp_history_job_details, temp_submit_job_details)
        except Exception as e:
            self.dispatcher_logger.error(f"sched_simulation_start error => {str(e)}")
            self.dispatcher_logger.exception(e)

    def sched_clear_all(self, ip, port):
        with get_zerorpc_client(ip, port) as client:
            client.clear_all()

    def stop_all(self, ip, port):
        with get_zerorpc_client(ip, port) as client:
            client.stop_all()

    def testbed_sched_dispatch_start(self, ip, port, 
                            cal_significance_sleep_time, 
                            scheduler_update_sleep_time, 
                            placement_sleep_time):
        with get_zerorpc_client(ip, port) as client:
            client.cal_significance_dispatch_start(cal_significance_sleep_time)
            client.sched_dispatch_start(scheduler_update_sleep_time)
            client.placement_dispatch_start(placement_sleep_time)
            client.thread_send_job_info_to_dispatcher_start()
    
    def sched_end(self, ip, port):
        try:
            with get_zerorpc_client(ip, port) as client:
                client.sched_end()
                all_result_map = client.end_and_report_dispatchers_by_sched()

            xlsx_save_path = "{}/{}/time_temp_log_{}.csv".format(RESULT_PATH, self.current_test_all_dir, self.simulation_index)
            new_xlsx_save_path = "{}/{}/time_log_{}.csv".format(RESULT_PATH, self.current_test_all_dir, self.simulation_index)
            os.rename(xlsx_save_path, new_xlsx_save_path)

            rm_list = []
            for _, _, files in os.walk(os.path.join(RESULT_PATH, self.current_test_all_dir)):
                for file in files:
                    if os.path.splitext(file)[1] == ".pt":
                        rm_list.append(os.path.basename(file))
            for rm_f in rm_list:
                os.remove(os.path.join(RESULT_PATH, self.current_test_all_dir, rm_f)) 

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
            self.dispatcher_logger.debug("all_test_jobs_num: {}".format(job_sequence_all_num))
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
        except Exception as e:
            self.dispatcher_logger.error(f"sched_end error => {str(e)}")
            self.dispatcher_logger.exception(e)

    def sched_report_status(self, ip, port, location):
        with get_zerorpc_client(ip, port) as client:
            client.report_status(location)

    def sched_update_gpu_status_start(self, ip, port, init_workerip_2_ports, init_gpuidentifiers, current_test_all_dir, simulation_index):
        with get_zerorpc_client(ip, port) as client:
            client.sched_update_gpu_status_start(init_workerip_2_ports, init_gpuidentifiers, current_test_all_dir, simulation_index)

    def sched_init_sched_register(self, ip, port, args, seed, 
                                assignment_policy, significance_policy, 
                                pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all,
                                config_max_operate_siton_run_num,
                                dataset_name, dataset_config_name, max_gpu_fuzai,
                                all_or_nothing_flag, enable_waiting_flag, inf_job_dispatch_flag, need_stop_lower_bound_ratio,
                                simulation, simulation_index):
        try:
            if assignment_policy == "PBGPolicy" or assignment_policy == "PBG":
                comparison_cost_epsilon_list = args.pbg_comparison_cost_epsilons
                comparison_z_threshold_list = args.pbg_comparison_z_thresholds
                L_list = args.pbg_Ls
                U_list = args.pbg_Us
                assignment_args = (pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all, comparison_cost_epsilon_list, comparison_z_threshold_list, L_list, U_list)
            elif assignment_policy == "PBGMixPolicy" or assignment_policy == "PBGMix":
                comparison_cost_epsilon_list = args.pbg_comparison_cost_epsilons
                comparison_z_threshold_list = args.pbg_comparison_z_thresholds
                L_list = args.pbg_Ls
                U_list = args.pbg_Us
                gitta_list = args.pbg_gittas
                assignment_args = (pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all, comparison_cost_epsilon_list, comparison_z_threshold_list, L_list, U_list, gitta_list)
            elif assignment_policy == "HISPolicy" or assignment_policy == "HIS" \
                or assignment_policy == "HISwithCPolicy" or assignment_policy == "HISwithC" \
                or assignment_policy == "HISwithOrderRemainVersionPolicy" or assignment_policy == "HISwithOrderRemainVersion" \
                or assignment_policy == "HISwithOrderProVersionPolicy" or assignment_policy == "HISwithOrderProVersion" \
                or assignment_policy == "HISwithOrderProVersionBestEffortPolicy" or assignment_policy == "HISwithOrderProVersionBestEffort":
                beta_list = args.his_betas
                infinity_flag = args.his_infinity_flag
                stop_n_growing_flag = args.his_stop_n_growing_flag
                greedy_flag = args.his_greedy_flag
                greedy_threshold = args.his_greedy_threshold
                assignment_args = (beta_list, pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all, infinity_flag, stop_n_growing_flag, greedy_flag, greedy_threshold)
            elif assignment_policy == "IterativeHISPolicy" or assignment_policy == "IterativeHIS" \
                or assignment_policy == "IterativeHISwithOrderProVersionPolicy" or assignment_policy == "IterativeHISwithOrderProVersion" \
                or assignment_policy == "IterativeHISwithOrderRemainVersionPolicy" or assignment_policy == "IterativeHISwithOrderRemainVersion" \
                or assignment_policy == "IterativeHISwithOrderProVersionBestEffortPolicy" or assignment_policy == "IterativeHISwithOrderProVersionBestEffort":
                beta_list = args.his_betas
                batch_size_for_one_epoch_list = args.his_batch_size_for_one_epochs
                infinity_flag = args.his_infinity_flag
                stop_n_growing_flag = args.his_stop_n_growing_flag
                greedy_flag = args.his_greedy_flag
                greedy_threshold = args.his_greedy_threshold
                adaptive_cons_generate_flag = args.his_adaptive_cons_generate_flag
                assignment_args = (beta_list, pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all, batch_size_for_one_epoch_list, infinity_flag, stop_n_growing_flag, greedy_flag, greedy_threshold, adaptive_cons_generate_flag)
            elif assignment_policy == "OfflinePolicy" or assignment_policy == "Offline" \
                or assignment_policy == "OfflineBestEffortPolicy" or assignment_policy == "OfflineBestEffort" \
                or assignment_policy == "SagewithRemainPolicy" or assignment_policy == "SagewithRemain" \
                or assignment_policy == "BestFitwithRemainPolicy" or assignment_policy == "BestFitwithRemain":
                assignment_args = (pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all)
            else:
                raise ValueError(f"assignment_policy: {assignment_policy} is abandoned!")
                assignment_args = None
            if significance_policy == "Temp" or significance_policy == "TempPolicy":
                significance_args = args.temp_sig_metric
            else:
                significance_args = ""
            
            with get_zerorpc_client(ip, port) as client:
                client.restart_sched()
                client.initialize_sched_configs(
                    simulation, 
                    simulation_index,
                    seed, 
                    self.current_test_all_dir, 
                    all_or_nothing_flag, 
                    enable_waiting_flag,
                    inf_job_dispatch_flag, 
                    need_stop_lower_bound_ratio,
                    pipeline_sequence_all_num,
                    job_request_all_num,
                    config_max_operate_siton_run_num,
                    dataset_name, 
                    dataset_config_name,
                    max_gpu_fuzai
                )
                client.sched_update_assignment_policy(assignment_policy, assignment_args)
                client.sched_update_significance_policy(significance_policy, significance_args)
                client.sched_update_current_time(self.all_start_time)
            self.dispatcher_logger.info("sched_init_sched_register finished!")
            log_args_var(args, self.dispatcher_logger_path)
        except Exception as e:
            self.dispatcher_logger.error(f"sched_init_sched_register error => {str(e)}")
            self.dispatcher_logger.exception(e)

def scheduler_listener_func(dispatcher_server_item):
    # def dispatcher_func_timely(dispatcher_server_item):
    # p = threading.Thread(target=dispatcher_func_timely, args=(dispatcher_server_item, ), daemon=True)
    # p.start()
    dispatcher_server = zerorpc.Server(dispatcher_server_item)
    ip_port = "tcp://0.0.0.0:{}".format(dispatcher_server_item.dispatcher_port)
    dispatcher_server.bind(ip_port)
    print("DL_server running in {}".format(ip_port))
    g = zerorpc.gevent.spawn(dispatcher_server.run)
    print("Thread [dispatcher_func_timely] started in !")
    return g

def exit_gracefully(server):
    print("stopping server")
    server.stop()
    print("closing server")
    server.close()



def testbed_experiment_start(
    args, sched_ip, sched_port,
    dispatcher_ip, dispatcher_port,
    worker_ips, worker_ports, 
    init_gpuidentifiers, 
    init_workerip_2_ports,
    global_sleep_time,
    scheduler_update_sleep_time, 
    cal_significance_sleep_time,
    placement_sleep_time, 
    waiting_time,
    dataset_reconstruct_path, 
    test_jobtrace_reconstruct_path, 
    history_jobtrace_reconstruct_path,
    dataset_name, 
    dataset_config_name,
    base_capacity, 
    job_dataset_trace_save_path, 
    current_test_all_dir, 
    restart_trace,
    pipeline_sequence_all_num, 
    all_history_num, 
    job_arrival_time_speed_up, 
    datablock_arrival_time_speed_up,
    all_datablock_num, 
    offline_datablock_num, 
    all_or_nothing_flag, 
    enable_waiting_flag,
    inf_job_dispatch_flag,
    need_stop_lower_bound_ratio,
    job_datablock_epsilon_max_ratio, 
    job_datablock_epsilon_min_ratio,
    job_require_select_block_min_num, 
    job_require_select_block_max_num,
    change_job_epsilon_max_times
):
    assert args.simulation_time == 1 and len(args.seeds) == 1
    simulation_flag = False
    valid_max_epsilon_require = base_capacity * job_datablock_epsilon_max_ratio
    valid_min_epsilon_require = base_capacity * job_datablock_epsilon_min_ratio
    offline_dispatch_job_dataset_once_time_flag = ("Offline" in args.assignment_policy) 
    
    offline_pipeline_sequence_all_num = 0
    offline_all_history_num = all_history_num
    if offline_dispatch_job_dataset_once_time_flag:
        print(f"offline_dispatch_job_dataset_once_time_flag: {offline_dispatch_job_dataset_once_time_flag} => change offline num")
        offline_datablock_num = all_datablock_num
        offline_pipeline_sequence_all_num = pipeline_sequence_all_num
        offline_all_history_num = all_history_num
        
    datasets_list, time_2_datablock_num = generate_alibaba_dataset(
        num=all_datablock_num,
        offline_num=offline_datablock_num,
        time_speed_up=datablock_arrival_time_speed_up,
        dataset_names=[dataset_name],
        fix_epsilon=base_capacity,
        fix_delta=1e-5,
        dataset_reconstruct_path=dataset_reconstruct_path, 
        save_path=job_dataset_trace_save_path
    )
    jobs_list = generate_alibaba_jobs(
        all_num=pipeline_sequence_all_num,
        offline_num=offline_pipeline_sequence_all_num,
        time_speed_up=job_arrival_time_speed_up,
        is_history=False,
        valid_max_epsilon_require=valid_max_epsilon_require,
        valid_min_epsilon_require=valid_min_epsilon_require,
        job_require_select_block_min_num=job_require_select_block_min_num,
        job_require_select_block_max_num=job_require_select_block_max_num,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        dispatcher_ip=dispatcher_ip,
        dispatcher_port=dispatcher_port,
        enable_waiting_flag=enable_waiting_flag,
        jobtrace_reconstruct_path=test_jobtrace_reconstruct_path,
        save_path=job_dataset_trace_save_path
    )
    history_jobs_list = generate_alibaba_jobs(
        all_num=all_history_num,
        offline_num=offline_all_history_num, # 默认就是离线的
        time_speed_up=job_arrival_time_speed_up,
        is_history=True,
        valid_max_epsilon_require=valid_max_epsilon_require,
        valid_min_epsilon_require=valid_min_epsilon_require,
        job_require_select_block_min_num=job_require_select_block_min_num,
        job_require_select_block_max_num=job_require_select_block_max_num,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        dispatcher_ip=dispatcher_ip,
        dispatcher_port=dispatcher_port,
        enable_waiting_flag=enable_waiting_flag,
        jobtrace_reconstruct_path=history_jobtrace_reconstruct_path,
        save_path=job_dataset_trace_save_path
    )
    pipeline_sequence_all_num = len(jobs_list)
    job_request_all_num = sys.maxsize if enable_waiting_flag else pipeline_sequence_all_num

    datablocks_privacy_budget_all = 0.0
    for _, datablocks in datasets_list.items():
        for _, datablock_info in datablocks.items():
            epsilon_capacity_per_datablock = datablock_info["epsilon_capacity"]
            datablocks_privacy_budget_all += epsilon_capacity_per_datablock
    print(f"datablocks_privacy_budget_all: {datablocks_privacy_budget_all}")

    processes = []
    dispatcher = Dispatcher(dispatcher_ip, dispatcher_port)
    dispatcher.restart_dispatcher(
        jobs_list=jobs_list, 
        history_jobs_list=history_jobs_list, 
        datasets_map=datasets_list, 
        current_test_all_dir=current_test_all_dir,
        simulation=False,
        simulation_index=0,
        restart_trace=restart_trace
    )
    
    remote_server_p = scheduler_listener_func(dispatcher)
    processes.append(remote_server_p)

    dispatcher.sched_init_sched_register(
        ip=sched_ip, port=sched_port, 
        args=args, seed=args.seeds[0], 
        assignment_policy=args.assignment_policy, 
        significance_policy=args.significance_policy, 
        pipeline_sequence_all_num=pipeline_sequence_all_num, 
        job_request_all_num=job_request_all_num,
        datablocks_privacy_budget_all=datablocks_privacy_budget_all,
        config_max_operate_siton_run_num=args.config_max_operate_siton_run_num,
        dataset_name=dataset_name, 
        dataset_config_name=dataset_config_name,
        max_gpu_fuzai=args.max_gpu_fuzai,
        all_or_nothing_flag=all_or_nothing_flag,
        enable_waiting_flag=enable_waiting_flag,
        inf_job_dispatch_flag=inf_job_dispatch_flag,
        need_stop_lower_bound_ratio=need_stop_lower_bound_ratio,
        simulation=False, simulation_index=0)
    dispatcher.sched_update_gpu_status_start(
        ip=sched_ip, port=sched_port, 
        init_workerip_2_ports=init_workerip_2_ports, 
        init_gpuidentifiers=init_gpuidentifiers, 
        current_test_all_dir=current_test_all_dir, 
        simulation_index=0
    )
    
    if not args.without_start_load_job:
        dataset_p = dispatcher.testbed_sched_update_dataset(sched_ip, sched_port)
        processes.append(dataset_p)
    
    zerorpc.gevent.sleep(waiting_time)
    time_p = dispatcher.sched_update_current_time()
    processes.append(time_p)

    if not args.without_start_load_history_job:
        history_job_p = dispatcher.testbed_dispatch_history_jobs(sched_ip, sched_port)
        processes.append(history_job_p)
    if not args.without_start_load_job:
        job_p = dispatcher.testbed_dispatch_jobs(inf_job_dispatch_flag, sched_ip, sched_port)
        processes.append(job_p)
    dispatcher.operator_job_results_start()

    print("Waiting for load datasets and jobs {} s".format(waiting_time))
    zerorpc.gevent.sleep(waiting_time)
    dispatcher.testbed_sched_dispatch_start(
        ip=sched_ip, port=sched_port, 
        cal_significance_sleep_time=cal_significance_sleep_time, 
        scheduler_update_sleep_time=scheduler_update_sleep_time, 
        placement_sleep_time=placement_sleep_time
    )
    
    # 主线程的最后一个操作!
    current_judge_key_job_ids = copy.deepcopy(dispatcher.need_judge_finished_label_keys)
    current_job_finished_labels = [dispatcher.finished_labels[job_id] for job_id in current_judge_key_job_ids]
    all_finished_label = reduce(lambda a, b: a and b, current_job_finished_labels)
    while not all_finished_label:
        zerorpc.gevent.sleep(global_sleep_time)
        current_judge_key_job_ids = copy.deepcopy(dispatcher.need_judge_finished_label_keys)
        current_job_finished_labels = [dispatcher.finished_labels[job_id] for job_id in current_judge_key_job_ids]
        all_finished_label = reduce(lambda a, b: a and b, current_job_finished_labels)
    dispatcher.sched_report_status(sched_ip, sched_port, "all stop")
    print("logically all stoped!")
    dispatcher.sched_end(sched_ip, sched_port)
    print("Stop workers and scheduler")
    zerorpc.gevent.sleep(waiting_time)
    if not args.without_finished_clear_all:
        dispatcher.sched_clear_all(sched_ip, sched_port)
    
    if not args.without_stop_all:
        dispatcher.stop_all(sched_ip, sched_port)
    all_result_file_name = "all_result.log"
    trace_save_dir_prefix = os.path.join(RESULT_PATH, current_test_all_dir)
    all_result_path = os.path.join(trace_save_dir_prefix, all_result_file_name)
    log_args_var(args, all_result_path)
    final_log_result(current_test_all_dir, all_result_file_name)
    print("Waiting for stop threads {} s".format(waiting_time))
    zerorpc.gevent.sleep(waiting_time)

def simulation_experiment_start(
    args, sched_ip, sched_port,
    dispatcher_ip, dispatcher_port,
    worker_ips, worker_ports, 
    init_gpuidentifiers, 
    init_workerip_2_ports,
    global_sleep_time,
    waiting_time,
    dataset_reconstruct_path, 
    test_jobtrace_reconstruct_path, 
    history_jobtrace_reconstruct_path,
    dataset_name, 
    dataset_config_name,
    base_capacity, 
    job_dataset_trace_save_path, 
    current_test_all_dir, 
    restart_trace,
    pipeline_sequence_all_num, 
    all_history_num, 
    job_arrival_time_speed_up, 
    datablock_arrival_time_speed_up,
    all_datablock_num, 
    offline_datablock_num, 
    simulation_time, 
    all_or_nothing_flag, 
    enable_waiting_flag,
    inf_job_dispatch_flag,
    need_stop_lower_bound_ratio,
    job_datablock_epsilon_max_ratio, 
    job_datablock_epsilon_min_ratio,
    job_require_select_block_min_num, 
    job_require_select_block_max_num,
    change_job_epsilon_max_times
):
    valid_max_epsilon_require = base_capacity * job_datablock_epsilon_max_ratio
    valid_min_epsilon_require = base_capacity * job_datablock_epsilon_min_ratio

    offline_dispatch_job_dataset_once_time_flag = ("Offline" in args.assignment_policy) 
    
    offline_pipeline_sequence_all_num = 0
    offline_all_history_num = all_history_num
    if offline_dispatch_job_dataset_once_time_flag:
        print(f"offline_dispatch_job_dataset_once_time_flag: {offline_dispatch_job_dataset_once_time_flag} => change offline num")
        offline_datablock_num = all_datablock_num
        offline_pipeline_sequence_all_num = pipeline_sequence_all_num
        offline_all_history_num = all_history_num

    datasets_list, time_2_datablock_num = generate_alibaba_dataset(
        num=all_datablock_num,
        offline_num=offline_datablock_num,
        time_speed_up=datablock_arrival_time_speed_up,
        dataset_names=[dataset_name],
        fix_epsilon=base_capacity,
        fix_delta=1e-5,
        dataset_reconstruct_path=dataset_reconstruct_path, 
        save_path=job_dataset_trace_save_path
    )
    jobs_list = generate_alibaba_jobs(
        all_num=pipeline_sequence_all_num,
        offline_num=offline_pipeline_sequence_all_num,
        time_speed_up=job_arrival_time_speed_up,
        is_history=False,
        valid_max_epsilon_require=valid_max_epsilon_require,
        valid_min_epsilon_require=valid_min_epsilon_require,
        job_require_select_block_min_num=job_require_select_block_min_num,
        job_require_select_block_max_num=job_require_select_block_max_num,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        dispatcher_ip=dispatcher_ip,
        dispatcher_port=dispatcher_port,
        enable_waiting_flag=enable_waiting_flag,
        jobtrace_reconstruct_path=test_jobtrace_reconstruct_path,
        save_path=job_dataset_trace_save_path
    )
    history_jobs_list = generate_alibaba_jobs(
        all_num=all_history_num,
        offline_num=offline_all_history_num, # 默认就是离线的
        time_speed_up=job_arrival_time_speed_up,
        is_history=True,
        valid_max_epsilon_require=valid_max_epsilon_require,
        valid_min_epsilon_require=valid_min_epsilon_require,
        job_require_select_block_min_num=job_require_select_block_min_num,
        job_require_select_block_max_num=job_require_select_block_max_num,
        change_job_epsilon_max_times=change_job_epsilon_max_times,
        dispatcher_ip=dispatcher_ip,
        dispatcher_port=dispatcher_port,
        enable_waiting_flag=enable_waiting_flag,
        jobtrace_reconstruct_path=history_jobtrace_reconstruct_path,
        save_path=job_dataset_trace_save_path
    )
    pipeline_sequence_all_num = len(jobs_list)
    job_request_all_num = sys.maxsize if enable_waiting_flag else pipeline_sequence_all_num

    datablocks_privacy_budget_all = 0.0
    for _, datablocks in datasets_list.items():
        for _, datablock_info in datablocks.items():
            epsilon_capacity_per_datablock = datablock_info["epsilon_capacity"]
            datablocks_privacy_budget_all += epsilon_capacity_per_datablock
    print(f"datablocks_privacy_budget_all: {datablocks_privacy_budget_all}")

    processes = []
    dispatcher = Dispatcher(dispatcher_ip, dispatcher_port)
    remote_server_p = scheduler_listener_func(dispatcher)
    processes.append(remote_server_p)
    for simulation_index in range(simulation_time):
        print("start simulation_index: {}".format(simulation_index))
        dispatcher.restart_dispatcher(
            jobs_list=jobs_list, 
            history_jobs_list=history_jobs_list, 
            datasets_map=datasets_list, 
            current_test_all_dir=current_test_all_dir, 
            simulation=True, 
            simulation_index=simulation_index,
            restart_trace=restart_trace
        )
        
        dispatcher.sched_init_sched_register(
            ip=sched_ip, port=sched_port, 
            args=args, seed=args.seeds[simulation_index], 
            assignment_policy=args.assignment_policy, significance_policy=args.significance_policy, 
            pipeline_sequence_all_num=pipeline_sequence_all_num, 
            job_request_all_num=job_request_all_num,
            datablocks_privacy_budget_all=datablocks_privacy_budget_all,
            config_max_operate_siton_run_num=args.config_max_operate_siton_run_num,
            dataset_name=dataset_name, 
            dataset_config_name=dataset_config_name,
            max_gpu_fuzai=args.max_gpu_fuzai,
            all_or_nothing_flag=all_or_nothing_flag,
            enable_waiting_flag=enable_waiting_flag,
            inf_job_dispatch_flag=inf_job_dispatch_flag, 
            need_stop_lower_bound_ratio=need_stop_lower_bound_ratio,
            simulation=True,
            simulation_index=simulation_index
        )
        dispatcher.sched_update_gpu_status_start(
            sched_ip, sched_port, 
            init_workerip_2_ports, init_gpuidentifiers, 
            current_test_all_dir, simulation_index
        )
        dispatcher.operator_job_results_start()
        # 合并成统一的事件队列, 需要等待所有的
        dispatcher.sched_simulation_start(sched_ip, sched_port) # 同样直接启动一个线程, 完成一大堆队列操作即可

        # 主线程的最后一个操作!
        current_judge_key_job_ids = copy.deepcopy(dispatcher.need_judge_finished_label_keys)
        current_job_finished_labels = [dispatcher.finished_labels[job_id] for job_id in current_judge_key_job_ids]
        all_finished_label = reduce(lambda a, b: a and b, current_job_finished_labels)
        while not all_finished_label:
            zerorpc.gevent.sleep(global_sleep_time)
            current_judge_key_job_ids = copy.deepcopy(dispatcher.need_judge_finished_label_keys)
            current_job_finished_labels = [dispatcher.finished_labels[job_id] for job_id in current_judge_key_job_ids]
            all_finished_label = reduce(lambda a, b: a and b, current_job_finished_labels)
        dispatcher.sched_report_status(sched_ip, sched_port, "all stop")
        print("logically all stoped!")
        dispatcher.sched_end(sched_ip, sched_port)
        if not args.without_finished_clear_all:
            dispatcher.sched_clear_all(sched_ip, sched_port)
        print("Stop workers and scheduler")
        zerorpc.gevent.sleep(waiting_time)
        print("end simulation_index: {}".format(simulation_index))
    if not args.without_stop_all:
        dispatcher.stop_all(sched_ip, sched_port)
    all_result_file_name = "all_result.log"
    trace_save_dir_prefix = os.path.join(RESULT_PATH, current_test_all_dir)
    all_result_path = os.path.join(trace_save_dir_prefix, all_result_file_name)
    final_log_result(current_test_all_dir, all_result_file_name)
    log_args_var(args, all_result_path)
    print("Waiting for stop threads {} s".format(waiting_time))
    zerorpc.gevent.sleep(waiting_time)

if __name__ == '__main__':
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

    if len(args.save_to_origin_trace_path) > 0:
        current_test_all_dir = args.save_to_origin_trace_path
        restart_trace = True
    else:
        logging_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        schedule_type = 'simulation' if args.simulation_flag else 'testbed'
        current_test_all_dir = 'schedule-review-%s-%s' % (schedule_type, logging_time)
        restart_trace = False
    job_dataset_trace_save_path = current_test_all_dir if args.need_save_jobtrace_flag else ""
    

    if args.simulation_flag:
        simulation_experiment_start(
            args=args, sched_ip=sched_ip, sched_port=sched_port,
            dispatcher_ip=dispatcher_ip, dispatcher_port=dispatcher_port,
            worker_ips=worker_ips, worker_ports=worker_ports, 
            init_gpuidentifiers=init_gpuidentifiers, 
            init_workerip_2_ports=init_workerip_2_ports,

            global_sleep_time=args.global_sleep_time,
            waiting_time=args.waiting_time, 

            dataset_reconstruct_path=args.dataset_reconstruct_path, 
            test_jobtrace_reconstruct_path=args.test_jobtrace_reconstruct_path, 
            history_jobtrace_reconstruct_path=args.history_jobtrace_reconstruct_path,
            dataset_name=args.dataset_name, 
            dataset_config_name=args.dataset_config_name,
            base_capacity=args.base_capacity, 
            job_dataset_trace_save_path=job_dataset_trace_save_path, 
            current_test_all_dir=current_test_all_dir, 
            restart_trace=restart_trace,
            pipeline_sequence_all_num=args.pipeline_sequence_all_num, 
            all_history_num=args.all_history_num, 
            job_arrival_time_speed_up=args.job_arrival_time_speed_up, 
            datablock_arrival_time_speed_up=args.datablock_arrival_time_speed_up,
            all_datablock_num=args.all_datablock_num, 
            offline_datablock_num=args.offline_datablock_num, 
            simulation_time=args.simulation_time,
            all_or_nothing_flag=args.all_or_nothing_flag, 
            enable_waiting_flag=args.enable_waiting_flag,
            inf_job_dispatch_flag=args.inf_job_dispatch_flag,
            need_stop_lower_bound_ratio=args.need_stop_lower_bound_ratio,
            job_datablock_epsilon_max_ratio=args.job_datablock_epsilon_max_ratio, 
            job_datablock_epsilon_min_ratio=args.job_datablock_epsilon_min_ratio,
            job_require_select_block_min_num=args.job_require_select_block_min_num, 
            job_require_select_block_max_num=args.job_require_select_block_max_num,
            change_job_epsilon_max_times=args.change_job_epsilon_max_times
        )
    else:
        testbed_experiment_start(
            args=args, sched_ip=sched_ip, sched_port=sched_port,
            dispatcher_ip=dispatcher_ip, dispatcher_port=dispatcher_port,
            worker_ips=worker_ips, worker_ports=worker_ports, 
            init_gpuidentifiers=init_gpuidentifiers, 
            init_workerip_2_ports=init_workerip_2_ports,
            
            global_sleep_time=args.global_sleep_time, 
            scheduler_update_sleep_time=args.scheduler_update_sleep_time,
            cal_significance_sleep_time=args.cal_significance_sleep_time, 
            placement_sleep_time=args.placement_sleep_time, 
            waiting_time=args.waiting_time,
            
            dataset_reconstruct_path=args.dataset_reconstruct_path, 
            test_jobtrace_reconstruct_path=args.test_jobtrace_reconstruct_path, 
            history_jobtrace_reconstruct_path=args.history_jobtrace_reconstruct_path,
            dataset_name=args.dataset_name, 
            dataset_config_name=args.dataset_config_name,
            base_capacity=args.base_capacity, 
            job_dataset_trace_save_path=job_dataset_trace_save_path, 
            current_test_all_dir=current_test_all_dir,
            restart_trace=restart_trace,
            pipeline_sequence_all_num=args.pipeline_sequence_all_num, 
            all_history_num=args.all_history_num, 
            job_arrival_time_speed_up=args.job_arrival_time_speed_up, 
            datablock_arrival_time_speed_up=args.datablock_arrival_time_speed_up,
            all_datablock_num=args.all_datablock_num, 
            offline_datablock_num=args.offline_datablock_num, 
            all_or_nothing_flag=args.all_or_nothing_flag, 
            enable_waiting_flag=args.enable_waiting_flag,
            inf_job_dispatch_flag=args.inf_job_dispatch_flag,
            need_stop_lower_bound_ratio=args.need_stop_lower_bound_ratio,
            job_datablock_epsilon_max_ratio=args.job_datablock_epsilon_max_ratio, 
            job_datablock_epsilon_min_ratio=args.job_datablock_epsilon_min_ratio,
            job_require_select_block_min_num=args.job_require_select_block_min_num, 
            job_require_select_block_max_num=args.job_require_select_block_max_num,
            change_job_epsilon_max_times=args.change_job_epsilon_max_times
        )    

    yag = yagmail.SMTP( user="xlc1220368815@163.com", password="PQVQLCNEICNYPFBM", host='smtp.163.com')
    contents = [
        f"[finished!] sched_ip: {sched_ip}; sched_port: {sched_port}; dispatcher_ip: {dispatcher_ip}; dispatcher_port: {dispatcher_port}; worker_ips: {worker_ips}; worker_ports: {worker_ports}; worker_indexes: {worker_indexes}"
    ]
    yag.send('1220368815@qq.com', 'subject', contents)
    
        