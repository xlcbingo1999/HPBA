import zerorpc
import time
import threading
import argparse
from utils.profier import timely_update_gpu_status
import torch
import json
import os
import sys

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
                            device_index, 
                            model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
                            LR, EPSILON, DELTA, MAX_GRAD_NORM, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, 
                            begin_epoch_num, run_epoch_num):
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

    execute_cmds.append("--device_index {}".format(device_index))
    if len(summary_writer_path) > 0 and len(summary_writer_key) > 0:
        execute_cmds.append("--summary_writer_path {}".format(summary_writer_path))
        execute_cmds.append("--summary_writer_key {}".format(summary_writer_key))
    if len(logging_file_path) > 0:
        execute_cmds.append("--logging_file_path {}".format(logging_file_path))
    if len(model_save_path) > 0:
        execute_cmds.append("--model_save_path {}".format(model_save_path))
    execute_cmds.append("--LR {}".format(LR))
    execute_cmds.append("--EPSILON {}".format(EPSILON))
    execute_cmds.append("--DELTA {}".format(DELTA))
    execute_cmds.append("--MAX_GRAD_NORM {}".format(MAX_GRAD_NORM))

    execute_cmds.append("--BATCH_SIZE {}".format(BATCH_SIZE))
    execute_cmds.append("--MAX_PHYSICAL_BATCH_SIZE {}".format(MAX_PHYSICAL_BATCH_SIZE))
    execute_cmds.append("--begin_epoch_num {}".format(begin_epoch_num))
    execute_cmds.append("--run_epoch_num {}".format(run_epoch_num))

    finally_execute_cmd = " ".join(execute_cmds)
    # print(finally_execute_cmd)
    print("Job {} start!".format(job_id))
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
        # gpu_device_count = torch.cuda.device_count()
        # self.worker_gpus_ready = {index:True for index in range(gpu_device_count)} # 直接允许即可

    def clear_all_jobs(self):
        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}
        print("success clear all jobs in worker!")

    def stop_all(self):
        self.all_finished = True

    def get_scheduler_zerorpc_client(self):
        tcp_ip_port = "tcp://{}:{}".format(self.sched_ip, self.sched_port)
        client = zerorpc.Client()
        client.connect(tcp_ip_port)
        return client

    def finished_job_callback(self, job_id, result, real_duration_time):
        origin_info = self.jobid_2_origininfo[job_id]
        self.report_result(job_id, origin_info, result, real_duration_time)
        tcp_ip_port = "tcp://{}:{}".format(self.sched_ip, self.sched_port)
        client = self.get_scheduler_zerorpc_client()
        client.worker_finished_job_callback(job_id, origin_info, result)
        del self.jobid_2_origininfo[job_id]
        del self.jobid_2_thread[job_id]

    def failed_job_callback(self, job_id, failed_result_key):
        origin_info = self.jobid_2_origininfo[job_id]
        client = self.get_scheduler_zerorpc_client()
        client.worker_failed_job_callback(job_id, origin_info, failed_result_key)
        if job_id in self.jobid_2_origininfo:
            del self.jobid_2_origininfo[job_id]
        if job_id in self.jobid_2_thread:
            self.jobid_2_thread[job_id].join()
            del self.jobid_2_thread[job_id]

    """
    def update_worker_gpus_status_callback(self, new_status_map):
        # 需要告知调度器, worker的gpu进行了更改, 同时更改worker的状态, 目前还是只考虑一起改变的情况吧
        for gpu_id in new_status_map:
            self.worker_gpus_ready[gpu_id] = new_status_map[gpu_id]
        client = self.get_scheduler_zerorpc_client()
        need_update_gpus_identifier = [("{}-{}".format(self.local_ip, gpu_id), gpu_id) for gpu_id in new_status_map]
        for worker_gpu_identifier, gpu_id in need_update_gpus_identifier:
            client.worker_gpu_status_callback(worker_gpu_identifier, new_status_map[gpu_id])
    """

    def begin_job(self, job_id, worker_gpu_id, worker_dataset_config, origin_info, sched_epsilon,
                  begin_epoch_num, update_sched_epoch_num, 
                  model_save_path, summary_writer_path, summary_writer_key, logging_file_path):
        # print("[bugxlc] job_id: {} call caculate => info: {}".format(job_id, origin_info))
        self.jobid_2_origininfo[job_id] = origin_info
        try:
            # GPU调度
            device_index = worker_gpu_id
            
            # DATASET调度
            train_dataset_name = worker_dataset_config["train_dataset_name"]
            test_dataset_name = worker_dataset_config["test_dataset_name"]
            sub_train_key_ids = worker_dataset_config["sub_train_key_ids"]
            sub_test_key_id = worker_dataset_config["sub_test_key_id"]

            model_name = origin_info["model_name"]
            
            LR = origin_info["LR"]
            EPSILON = sched_epsilon
            DELTA = origin_info["DELTA"]
            MAX_GRAD_NORM = origin_info["MAX_GRAD_NORM"]
            BATCH_SIZE = origin_info["BATCH_SIZE"]
            MAX_PHYSICAL_BATCH_SIZE = origin_info["MAX_PHYSICAL_BATCH_SIZE"]

            worker_ip = self.local_ip
            worker_port = self.local_port

            p = threading.Thread(target=do_system_calculate_func, args=(worker_ip, worker_port, 
                job_id, model_name, 
                train_dataset_name, test_dataset_name,
                sub_train_key_ids, sub_test_key_id, 
                device_index, 
                model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
                LR, EPSILON, DELTA, MAX_GRAD_NORM, 
                BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, begin_epoch_num, update_sched_epoch_num), daemon=True)
            self.jobid_2_thread[job_id] = p
            p.start()
        except Exception as e:
            self.failed_job_callback(job_id, "FAILED_RESULT_KEY.JOB_FAILED")
            raise ValueError("No this calculate func: {}".format(e))

    def report_result(self, job_id, origin_info, result, real_duration_time):
        print("Worker finished job [{}] => result: {}; time: {}".format(job_id, result, real_duration_time))

    def timely_update_gpu_status(self):
        gpu_devices_count = torch.cuda.device_count()
        dids = range(gpu_devices_count)
        p = threading.Thread(target=timely_update_gpu_status, args=(self.local_ip, dids, self.gpu_update_time), daemon=True)
        p.start()


def worker_listener_func(worker_server_item):
    def work_func_timely(worker_server_item):
        s = zerorpc.Server(worker_server_item)
        ip_port = "tcp://0.0.0.0:{}".format(worker_server_item.local_port)
        s.bind(ip_port)
        print("DL_server running in {}".format(ip_port))
        s.run()
    p = threading.Thread(target=work_func_timely, args=(worker_server_item, ), daemon=True)
    p.start()
    return p

if __name__ == "__main__":
    args = get_df_config()
    local_ip, local_port, sched_ip, sched_port, gpu_update_time = args.local_ip, args.local_port, args.sched_ip, args.sched_port, args.gpu_update_time
    worker_server_item = Worker_server(local_ip, local_port, sched_ip, sched_port, gpu_update_time)
    # worker_server_item.timely_update_gpu_status()
    worker_p = worker_listener_func(worker_server_item)

    while not worker_server_item.all_finished:
        time.sleep(10)
    print("DL sched finished!!")
    
    sys.exit(0)