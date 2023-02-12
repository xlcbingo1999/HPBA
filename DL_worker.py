import zerorpc
import time
import threading
import argparse
from utils.global_functions import FAILED_RESULT_KEY
from utils.global_variable import WORKER_LOCAL_IP, WORKER_LOCAL_PORT, SCHE_IP, SCHE_PORT, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH
import torch
import json
import os

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--local_ip", type=str, default=WORKER_LOCAL_IP)
    parser.add_argument("--local_port", type=int, default=WORKER_LOCAL_PORT)
    parser.add_argument("--sched_ip", type=str, default=SCHE_IP)
    parser.add_argument("--sche_port", type=int, default=SCHE_PORT)

    args = parser.parse_args()
    return args

def do_system_calculate_func(worker_ip, worker_port, job_id, model_name, train_dataset_raw_paths, test_dataset_raw_path,
                            dataset_name, label_type, selected_datablock_identifiers, not_selected_datablock_identifiers,
                            device, early_stopping, summary_writer_path,
                            LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                            label_distributions, train_configs):
    execute_cmds = []
    execute_cmds.append("conda run -n py39torch113")
    execute_cmds.append("python DL_do_calculate.py")
    execute_cmds.append("--worker_ip {}".format(worker_ip))
    execute_cmds.append("--worker_port {}".format(worker_port))
    execute_cmds.append("--job_id {}".format(job_id))
    execute_cmds.append("--model_name {}".format(model_name))
    train_dataset_raw_paths_str = ":".join(train_dataset_raw_paths)
    execute_cmds.append("--train_dataset_raw_paths {}".format(train_dataset_raw_paths_str))
    execute_cmds.append("--test_dataset_raw_path {}".format(test_dataset_raw_path))
    execute_cmds.append("--dataset_name {}".format(dataset_name))
    execute_cmds.append("--label_type {}".format(label_type))
    selected_datablock_identifiers_str = ":".join(selected_datablock_identifiers)
    execute_cmds.append("--selected_datablock_identifiers {}".format(selected_datablock_identifiers_str))
    not_selected_datablock_identifiers_str = ":".join(not_selected_datablock_identifiers)
    execute_cmds.append("--not_selected_datablock_identifiers {}".format(not_selected_datablock_identifiers_str))

    execute_cmds.append("--device {}".format(device))
    if early_stopping:
        execute_cmds.append("--early_stopping")
    if len(summary_writer_path) > 0:
        execute_cmds.append("--summary_writer_path {}".format(summary_writer_path))
    execute_cmds.append("--LR {}".format(LR))
    execute_cmds.append("--EPSILON {}".format(EPSILON))
    if EPOCH_SET_EPSILON:
        execute_cmds.append("--EPOCH_SET_EPSILON {}".format(EPOCH_SET_EPSILON))
    execute_cmds.append("--DELTA {}".format(DELTA))
    execute_cmds.append("--MAX_GRAD_NORM {}".format(MAX_GRAD_NORM))

    execute_cmds.append("--BATCH_SIZE {}".format(BATCH_SIZE))
    execute_cmds.append("--MAX_PHYSICAL_BATCH_SIZE {}".format(MAX_PHYSICAL_BATCH_SIZE))
    execute_cmds.append("--EPOCHS {}".format(EPOCHS))
    label_distributions_str = json.dumps(label_distributions)
    # print("[label_distributions_str] dump result: ", label_distributions_str)
    execute_cmds.append("--label_distributions '{}'".format(label_distributions_str))
    train_configs_str = json.dumps(train_configs)
    # print("[train_configs_str] dump result: ", train_configs_str)
    execute_cmds.append("--train_configs '{}'".format(train_configs_str))

    finally_execute_cmd = " ".join(execute_cmds)
    print(finally_execute_cmd)
    os.system(finally_execute_cmd)


class Worker_server(object):
    def __init__(self, local_ip, local_port, sched_ip, sche_port):
        # self.local_worker_id = None
        self.local_ip = local_ip
        self.local_port = local_port
        self.sched_ip = sched_ip
        self.sche_port = sche_port
        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}
        # gpu_device_count = torch.cuda.device_count()
        # self.worker_gpus_ready = {index:True for index in range(gpu_device_count)} # 直接允许即可

    def clear_all_jobs(self):
        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}
        print("success clear all jobs in worker!")

    def get_scheduler_zerorpc_client(self):
        tcp_ip_port = "tcp://{}:{}".format(self.sched_ip, self.sche_port)
        client = zerorpc.Client()
        client.connect(tcp_ip_port)
        return client

    def finished_job_callback(self, job_id, result, real_duration_time):
        origin_info = self.jobid_2_origininfo[job_id]
        self.report_result(job_id, origin_info, result, real_duration_time)
        tcp_ip_port = "tcp://{}:{}".format(self.sched_ip, self.sche_port)
        client = self.get_scheduler_zerorpc_client()
        client.worker_finished_job_callback(job_id, origin_info, result)
        del self.jobid_2_origininfo[job_id]
        del self.jobid_2_thread[job_id]

    def failed_job_callback(self, job_id, failed_result_key):
        client = self.get_scheduler_zerorpc_client()
        client.worker_failed_job_callback(job_id, failed_result_key)
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

    def begin_job(self, job_id, worker_gpu_id, worker_dataset_config, origin_info):
        # print("[bugxlc] job_id: {} call caculate => info: {}".format(job_id, origin_info))
        self.jobid_2_origininfo[job_id] = origin_info
        target_func = origin_info["target_func"]
        try:
            if target_func == "opacus_split_review":
                # GPU调度
                device = worker_gpu_id
                
                # DATASET调度
                selected_datablock_identifiers = worker_dataset_config["selected_datablock_identifiers"]
                not_selected_datablock_identifiers = worker_dataset_config["not_selected_datablock_identifiers"]
                train_dataset_raw_paths = worker_dataset_config["train_dataset_raw_paths"]
                test_dataset_raw_path = worker_dataset_config["test_dataset_raw_path"]
                label_distributions = worker_dataset_config["label_distributions"]

                dataset_name = origin_info["dataset_name"]
                label_type = origin_info["label_type"]
                model_name = origin_info["model_name"]
                early_stopping = origin_info["early_stopping"]
                train_configs = origin_info["train_configs"]
                
                LR = origin_info["LR"]
                EPSILON = origin_info["EPSILON"]
                EPOCH_SET_EPSILON = origin_info["EPOCH_SET_EPSILON"]
                DELTA = origin_info["DELTA"]
                MAX_GRAD_NORM = origin_info["MAX_GRAD_NORM"]
                BATCH_SIZE = origin_info["BATCH_SIZE"]
                MAX_PHYSICAL_BATCH_SIZE = origin_info["MAX_PHYSICAL_BATCH_SIZE"]
                EPOCHS = origin_info["EPOCHS"]

                summary_writer_path = ""
                worker_ip = self.local_ip
                worker_port = self.local_port

                p = threading.Thread(target=do_system_calculate_func, args=(worker_ip, worker_port,
                                                                    job_id, model_name, train_dataset_raw_paths, test_dataset_raw_path,
                                                                    dataset_name, label_type, selected_datablock_identifiers, not_selected_datablock_identifiers,
                                                                    device, early_stopping, summary_writer_path,
                                                                    LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                                                                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                                                                    label_distributions,
                                                                    train_configs), daemon=True)
                self.jobid_2_thread[job_id] = p
                p.start()
            else:
                self.failed_job_callback(job_id, FAILED_RESULT_KEY.JOB_TYPE_ERROR)
                raise ValueError("No this calculate func: {}".format(target_func))
        except Exception as e:
            self.failed_job_callback(job_id, FAILED_RESULT_KEY.JOB_FAILED)
            raise ValueError("No this calculate func: {}".format(e))

    def report_result(self, job_id, origin_info, result, real_duration_time):
        print("Worker finished job [{}] => result: {}; time: {}".format(job_id, result, real_duration_time))


def worker_listener_func(worker_server_item):
    s = zerorpc.Server(worker_server_item)
    ip_port = "tcp://0.0.0.0:{}".format(worker_server_item.local_port)
    s.bind(ip_port)
    print("DL_server running in {}".format(ip_port))
    s.run()

if __name__ == "__main__":
    args = get_df_config()
    local_ip, local_port, sched_ip, sche_port = args.local_ip, args.local_port, args.sched_ip, args.sche_port
    worker_server_item = Worker_server(local_ip, local_port, sched_ip, sche_port)
    worker_listener_func(worker_server_item)
    
