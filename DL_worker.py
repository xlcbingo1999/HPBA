import zerorpc
import time
import threading
import argparse
from opacus_scheduler_job import do_calculate_func
from utils.data_loader import fetch_new_dataset
from utils.global_functions import FAILED_RESULT_KEY
from utils.global_variable import WORKER_LOCAL_IP, WORKER_LOCAL_PORT, SCHE_IP, SCHE_PORT, MAX_EPSILON

import torch

def get_df_config():
    parser = argparse.ArgumentParser(
                description='Sweep through lambda values')
    parser.add_argument('--local_ip', type=str, default=WORKER_LOCAL_IP)
    parser.add_argument('--local_port', type=int, default=WORKER_LOCAL_PORT)
    parser.add_argument('--sched_ip', type=str, default=SCHE_IP)
    parser.add_argument('--sche_port', type=int, default=SCHE_PORT)

    args = parser.parse_args()
    return args

def long_time_add(job_id, x, y, callback):    
    time.sleep(5)
    callback(job_id, x + y)

class Worker_server(object):
    def __init__(self, local_ip, local_port, sched_ip, sche_port):
        # self.local_worker_id = None
        self.local_ip = local_ip
        self.local_port = local_port
        self.sched_ip = sched_ip
        self.sche_port = sche_port
        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}

        # self.datasets = {} # 考虑未来引入多种数据集
        self.sub_train_datasets = None
        self.valid_dataset = None
        self.output_size = None
        self.vocab_size = None
        self.summary_writer = None # TODO(xlc): 之后需要将summary_writer写入

        self.worker_dataset_ready = False
        gpu_device_count = torch.cuda.device_count()
        self.worker_gpus_ready = {index:True for index in range(gpu_device_count)} # 直接允许即可

    def clear_all_jobs(self):
        self.jobid_2_origininfo = {}
        self.jobid_2_thread = {}

    # 底层的数据直接共享, sub_train_datasets和valid_dataset应该由调度器来决定内容, 考虑从外部注入!
    # 本操作应该是一个同步操作, 只有所有的worker都load到了数据, 才可以认为worker可用. 故需要增加worker状态的标记
    def initial_dataset(self, fetch_dataset_origin_info, keep_origin_dataset):
        if keep_origin_dataset and self.worker_dataset_ready:
            print("worker load dataset success! [Warning: keep_origin_dataset]")
            return
        self.update_worker_dataset_status_callback(False)
        DATASET_NAME = fetch_dataset_origin_info['DATASET_NAME']
        LABEL_TYPE = fetch_dataset_origin_info['LABEL_TYPE']
        VALID_SIZE = fetch_dataset_origin_info['VALID_SIZE']
        SEQUENCE_LENGTH = fetch_dataset_origin_info['SEQUENCE_LENGTH']
        # BATCH_SIZE = fetch_dataset_origin_info[] # TODO(xlc): 这里明显有问题, 不应该把BATCH_SIZE这种内容在这里传
        SPLIT_NUM = fetch_dataset_origin_info['SPLIT_NUM']
        # ALPHA = fetch_dataset_origin_info[]
        same_capacity = fetch_dataset_origin_info['same_capacity']
        # plot_path = fetch_dataset_origin_info[]
        _, sub_train_datasets, valid_dataset, \
            output_size, vocab_size = fetch_new_dataset(DATASET_NAME, LABEL_TYPE, VALID_SIZE, SEQUENCE_LENGTH, SPLIT_NUM, same_capacity)

        self.sub_train_datasets = sub_train_datasets
        self.valid_dataset = valid_dataset
        self.output_size = output_size
        self.vocab_size = vocab_size
        print("worker load dataset success!")

        self.update_worker_dataset_status_callback(True)        
        

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
        origin_info = self.jobid_2_origininfo[job_id]
        client = self.get_scheduler_zerorpc_client()
        client.worker_failed_job_callback(job_id, failed_result_key)
        if job_id in self.jobid_2_origininfo:
            del self.jobid_2_origininfo[job_id]
        if job_id in self.jobid_2_thread:
            del self.jobid_2_thread[job_id]
        
    def update_worker_dataset_status_callback(self, new_status):
        # 需要告知调度器, worker的dataset进行了更改, 同时更改worker的状态
        self.worker_dataset_ready = new_status
        client = self.get_scheduler_zerorpc_client()
        client.worker_dataset_status_callback(self.local_ip, self.worker_dataset_ready)

    def update_worker_gpus_status_callback(self, new_status_map):
        # 需要告知调度器, worker的gpu进行了更改, 同时更改worker的状态, 目前还是只考虑一起改变的情况吧
        for gpu_id in new_status_map:
            self.worker_gpus_ready[gpu_id] = new_status_map[gpu_id]
        client = self.get_scheduler_zerorpc_client()
        need_update_gpus_identifier = [("{}-{}".format(self.local_ip, gpu_id), gpu_id) for gpu_id in new_status_map]
        for worker_gpu_identifier, gpu_id in need_update_gpus_identifier:
            client.worker_gpu_status_callback(worker_gpu_identifier, new_status_map[gpu_id])

    def begin_job(self, job_id, worker_gpu_id, worker_dataset_config, origin_info):
        print("[bugxlc] job_id: {} call caculate => info: {}".format(job_id, origin_info))
        if not self.worker_dataset_ready:
            self.failed_job_callback(job_id, FAILED_RESULT_KEY.WORKER_NO_READY)
            return
        self.jobid_2_origininfo[job_id] = origin_info
        target_func = origin_info['target_func']
        if target_func == "opacus_split_review":
            # GPU调度
            device = worker_gpu_id
            
            # DATASET调度
            is_select = worker_dataset_config['is_select']
            selected_datablock_ids = worker_dataset_config['selected_datablock_ids']
            not_selected_datablock_ids = worker_dataset_config['not_selected_datablock_ids']
            label_distributions = worker_dataset_config['label_distributions']

            model_name = origin_info['model_name']
            early_stopping = origin_info['early_stopping']
            train_configs = origin_info['train_configs']
            
            LR = origin_info['LR']
            EPSILON = origin_info['EPSILON']
            EPOCH_SET_EPSILON = origin_info['EPOCH_SET_EPSILON']
            DELTA = origin_info['DELTA']
            MAX_GRAD_NORM = origin_info['MAX_GRAD_NORM']
            BATCH_SIZE = origin_info['BATCH_SIZE']
            MAX_PHYSICAL_BATCH_SIZE = origin_info['MAX_PHYSICAL_BATCH_SIZE']
            EPOCHS = origin_info['EPOCHS']

            sub_train_datasets = self.sub_train_datasets # origin_info['sub_train_datasets']
            valid_dataset = self.valid_dataset # origin_info['valid_dataset']
            vocab_size = self.vocab_size # origin_info['vocab_size']
            output_size = self.output_size # origin_info['output_size']
            summary_writer = self.summary_writer

            p = threading.Thread(target=do_calculate_func, args=(job_id, model_name, is_select, 
                                                                sub_train_datasets, valid_dataset,
                                                                selected_datablock_ids, not_selected_datablock_ids, 
                                                                device, early_stopping, summary_writer,
                                                                LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                                                                BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, MAX_EPSILON, EPOCHS,
                                                                vocab_size, output_size, label_distributions,
                                                                train_configs, self.finished_job_callback))
            self.jobid_2_thread[job_id] = p
            p.start()
        else:
            self.failed_job_callback(job_id, FAILED_RESULT_KEY.JOB_TYPE_ERROR)
            raise ValueError("No this calculate func: {}".format(target_func))

    def report_result(self, job_id, origin_info, result, real_duration_time):
        print("=========  Worker  ===========")
        print("job_id: {}".format(job_id))
        print("origin_info: {}".format(origin_info))
        print("result: {}".format(result))
        print("real_duration_time: {}".format(real_duration_time))
        print("====================")


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
    
