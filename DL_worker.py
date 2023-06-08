import zerorpc
import time
import threading
import multiprocessing
import argparse
from utils.profier import timely_update_gpu_status
import torch
import json
import os
import sys
from utils.global_variable import RESULT_PATH
from utils.global_functions import get_zerorpc_client, print_console_file
from utils.logging_tools import get_logger

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--local_ip", type=str, default="172.18.162.6")
    parser.add_argument("--local_port", type=int, default=16206)
    parser.add_argument("--sched_ip", type=str, default="172.18.162.6")
    parser.add_argument("--sched_port", type=int, default=16306)
    
    parser.add_argument("--gpu_update_time", type=int, default=1)
    
    args = parser.parse_args()
    return args

def do_system_calculate_func(worker_ip, worker_port, 
                            job_id, model_name, 
                            train_dataset_name, test_dataset_name,
                            sub_train_key_ids, sub_test_key_id, 
                            sub_train_dataset_config_path, test_dataset_config_path,
                            device_index, 
                            model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
                            LR, EPSILON_one_siton, DELTA, MAX_GRAD_NORM, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, 
                            begin_epoch_num, siton_run_epoch_num, final_significance, 
                            simulation_flag, worker_logging_path):
    execute_cmds = []
    # execute_cmds.append("conda run -n py39torch113") # 会卡住, 没有任何log, 这个时候最好做重定向!
    execute_cmds.append("python -u DL_do_calculate.py")
    execute_cmds.append("--worker_ip {}".format(worker_ip))
    execute_cmds.append("--worker_port {}".format(worker_port))
    execute_cmds.append("--job_id {}".format(job_id))
    execute_cmds.append("--model_name {}".format(model_name))

    execute_cmds.append("--train_dataset_name {}".format(train_dataset_name))
    execute_cmds.append("--test_dataset_name {}".format(test_dataset_name))
    sub_train_key_ids = ":".join(sub_train_key_ids)
    execute_cmds.append("--sub_train_key_ids {}".format(sub_train_key_ids))
    execute_cmds.append("--sub_test_key_id {}".format(sub_test_key_id))

    execute_cmds.append("--sub_train_dataset_config_path {}".format(sub_train_dataset_config_path))
    execute_cmds.append("--test_dataset_config_path {}".format(test_dataset_config_path))
    
    

    execute_cmds.append("--device_index {}".format(device_index))
    if len(summary_writer_path) > 0 and len(summary_writer_key) > 0:
        execute_cmds.append("--summary_writer_path {}".format(summary_writer_path))
        execute_cmds.append("--summary_writer_key {}".format(summary_writer_key))
    if len(logging_file_path) > 0:
        execute_cmds.append("--logging_file_path {}".format(logging_file_path))
    if len(model_save_path) > 0:
        execute_cmds.append("--model_save_path {}".format(model_save_path))
    execute_cmds.append("--LR {}".format(LR))
    execute_cmds.append("--EPSILON_one_siton {}".format(EPSILON_one_siton))
    execute_cmds.append("--DELTA {}".format(DELTA))
    execute_cmds.append("--MAX_GRAD_NORM {}".format(MAX_GRAD_NORM))

    execute_cmds.append("--BATCH_SIZE {}".format(BATCH_SIZE))
    execute_cmds.append("--MAX_PHYSICAL_BATCH_SIZE {}".format(MAX_PHYSICAL_BATCH_SIZE))
    execute_cmds.append("--begin_epoch_num {}".format(begin_epoch_num))
    execute_cmds.append("--siton_run_epoch_num {}".format(siton_run_epoch_num))
    execute_cmds.append("--final_significance {}".format(final_significance))
    if simulation_flag:
        execute_cmds.append("--simulation_flag")

    finally_execute_cmd = " ".join(execute_cmds)
    if len(worker_logging_path) > 0:
        with open(worker_logging_path, "a+") as f:
            print_console_file(finally_execute_cmd, fileHandler=f)
            print_console_file(f"Job {job_id} start!", fileHandler=f)
    os.system(finally_execute_cmd)


class Worker_server(object):
    def __init__(self, local_ip, local_port, sched_ip, sched_port, gpu_update_time):
        # self.local_worker_id = None
        self.local_ip = local_ip
        self.local_port = local_port
        self.sched_ip = sched_ip
        self.sched_port = sched_port

        self.gpu_update_time = gpu_update_time

        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}

        self.all_finished = False

        self.logger_path = ""
        self.worker_logger = None
        # gpu_device_count = torch.cuda.device_count()
        # self.worker_gpus_ready = {index:True for index in range(gpu_device_count)} # 直接允许即可

        self.failed_job_callback_thread = None
        self.failed_job_callback_list = []
        self.finished_job_callback_thread = None
        self.finished_job_callback_list = []

    def clear_all_jobs(self):
        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}
        self.failed_job_callback_thread = None
        self.failed_job_callback_list = []
        self.finished_job_callback_thread = None
        self.finished_job_callback_list = []
        self.worker_logger.info("success clear all jobs in worker!")

    def stop_all(self):
        print(f"worker {self.local_ip}:{self.local_port} stop_all")
        self.all_finished = True

    def finished_job_callback(self, job_id, result, real_duration_time):
        origin_info = self.jobid_2_origininfo[job_id]
        self.worker_logger.info("Worker finished job [{}] => result: {}; time: {}".format(job_id, result, real_duration_time))
        self.finished_job_callback_list.append({
            "job_id": job_id,
            "origin_info": origin_info,
            "result": result
        })
        if job_id in self.jobid_2_origininfo:
            del self.jobid_2_origininfo[job_id]
        if job_id in self.jobid_2_thread:
            del self.jobid_2_thread[job_id]

    def runtime_failed_job_callback(self, job_id, exception_log):
        origin_info = self.jobid_2_origininfo[job_id]
        self.failed_job_callback_list.append({
            "job_id": job_id,
            "origin_info": origin_info,
            "exception_log": exception_log
        })
        if job_id in self.jobid_2_origininfo:
            del self.jobid_2_origininfo[job_id]
        if job_id in self.jobid_2_thread:
            del self.jobid_2_thread[job_id]

    def runtime_failed_job_callback_start(self):
        def thread_func_timely_runtime_failed_job_callback(sleep_time):
            while not self.all_finished:
                while len(self.failed_job_callback_list) > 0:
                    details = self.failed_job_callback_list.pop(0)
                    job_id = details["job_id"]
                    origin_info = details["origin_info"]
                    exception_log = details["exception_log"]
                    client = get_zerorpc_client(self.sched_ip, self.sched_port)
                    client.worker_runtime_failed_job_callback(job_id, origin_info, exception_log)
                zerorpc.gevent.sleep(sleep_time)
            print("Thread thread_func_timely_runtime_failed_job_callback finished!")
        p = threading.Thread(target=thread_func_timely_runtime_failed_job_callback, args=(1,), daemon=True)
        self.failed_job_callback_thread = p
        p.start()
        print("Thread thread_func_timely_runtime_failed_job_callback start!")

    def finished_job_callback_start(self):
        def thread_func_timely_finished_job_callback(sleep_time):
            while not self.all_finished:
                while len(self.finished_job_callback_list) > 0:
                    details = self.finished_job_callback_list.pop(0)
                    job_id = details["job_id"]
                    origin_info = details["origin_info"]
                    result = details["result"]
                    client = get_zerorpc_client(self.sched_ip, self.sched_port)
                    client.worker_finished_job_callback(job_id, origin_info, result)
                zerorpc.gevent.sleep(sleep_time)
            print("Thread thread_func_timely_finished_job_callback finished!")
        p = threading.Thread(target=thread_func_timely_finished_job_callback, args=(1,), daemon=True)
        self.finished_job_callback_thread = p
        p.start()
        print("Thread thread_func_timely_finished_job_callback start!")

    def initialize_logging_path(self, current_test_all_dir, simulation_index):
        self.logger_path = "{}/{}/DL_worker_{}_{}_{}.log".format(RESULT_PATH, current_test_all_dir, self.local_ip, self.local_port, simulation_index) 
        self.worker_logger = get_logger(self.logger_path, self.logger_path, enable_multiprocess=True)

    def begin_job(self, job_id, worker_gpu_id, worker_dataset_config, origin_info, 
                  sched_epsilon_one_siton_run, begin_epoch_num, siton_run_epoch_num, 
                  model_save_path, summary_writer_path, summary_writer_key, logging_file_path, final_significance, simulation_flag):
        # self.worker_logger.info("[bugxlc] job_id: {} call caculate => info: {}".format(job_id, origin_info))
        self.jobid_2_origininfo[job_id] = origin_info
        if simulation_flag:
            all_results = {
                'train_acc': 0.0,
                'train_loss': 0.0,
                'test_acc': 0.0,
                'test_loss': 0.0,
                'epsilon_consume': sched_epsilon_one_siton_run,
                'begin_epoch_num': begin_epoch_num,
                'run_epoch_num': siton_run_epoch_num,
                'final_significance': final_significance
            }
            self.finished_job_callback(job_id, all_results, 0.0)
            return     
        device_index = worker_gpu_id
        
        train_dataset_name = worker_dataset_config["train_dataset_name"]
        test_dataset_name = worker_dataset_config["test_dataset_name"]
        sub_train_key_ids = worker_dataset_config["sub_train_key_ids"]
        sub_test_key_id = worker_dataset_config["sub_test_key_id"]
        sub_train_dataset_config_path = worker_dataset_config["sub_train_dataset_config_path"] 
        test_dataset_config_path = worker_dataset_config["test_dataset_config_path"]

        model_name = origin_info["model_name"]
        
        LR = origin_info["LR"]
        EPSILON_one_siton = sched_epsilon_one_siton_run
        DELTA = origin_info["DELTA"]
        MAX_GRAD_NORM = origin_info["MAX_GRAD_NORM"]
        BATCH_SIZE = origin_info["BATCH_SIZE"]
        MAX_PHYSICAL_BATCH_SIZE = origin_info["MAX_PHYSICAL_BATCH_SIZE"]
        self.worker_logger.info("EPSILON_one_siton in begin_job {}: [{}]".format(job_id, EPSILON_one_siton))

        worker_ip = self.local_ip
        worker_port = self.local_port

        p = threading.Thread(target=do_system_calculate_func, args=(worker_ip, worker_port, 
            job_id, model_name, 
            train_dataset_name, test_dataset_name,
            sub_train_key_ids, sub_test_key_id, 
            sub_train_dataset_config_path, test_dataset_config_path,
            device_index, 
            model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
            LR, EPSILON_one_siton, DELTA, MAX_GRAD_NORM, 
            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, begin_epoch_num, siton_run_epoch_num, final_significance, 
            simulation_flag, self.logger_path), daemon=True)
        self.jobid_2_thread[job_id] = p
        p.start()

    def timely_update_gpu_status(self):
        gpu_devices_count = torch.cuda.device_count()
        dids = range(gpu_devices_count)
        p = threading.Thread(target=timely_update_gpu_status, args=(self.local_ip, dids, self.gpu_update_time), daemon=True)
        p.start()


def worker_listener_func(worker_server_item):
    # def work_func_timely(worker_server_item):
    #     s = zerorpc.Server(worker_server_item)
    #     ip_port = "tcp://0.0.0.0:{}".format(worker_server_item.local_port)
    #     s.bind(ip_port)
    #     print("DL_server running in {}".format(ip_port))
    #     s.run()
    # p = threading.Thread(target=work_func_timely, args=(worker_server_item, ), daemon=True)
    # p.start()
    s = zerorpc.Server(worker_server_item)
    ip_port = "tcp://0.0.0.0:{}".format(worker_server_item.local_port)
    s.bind(ip_port)
    print("DL_server running in {}".format(ip_port))
    g = zerorpc.gevent.spawn(s.run)  
    return g

if __name__ == "__main__":
    args = get_df_config()
    local_ip, local_port, sched_ip, sched_port, gpu_update_time = args.local_ip, args.local_port, args.sched_ip, args.sched_port, args.gpu_update_time
    worker_server_item = Worker_server(local_ip, local_port, sched_ip, sched_port, gpu_update_time)
    # worker_server_item.timely_update_gpu_status()
    worker_p = worker_listener_func(worker_server_item)

    worker_server_item.finished_job_callback_start()
    worker_server_item.runtime_failed_job_callback_start()

    while not worker_server_item.all_finished:
        zerorpc.gevent.sleep(10)
    print("DL sched finished!!")
    
    sys.exit(0)