import zerorpc
import time
from utils.global_variable import SCHE_IP, SCHE_PORT, DISPATCHER_IP, DISPATCHER_PORT, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH, INIT_WORKERIDENTIFIERS, TENSORBOARD_PATH
import threading
from functools import reduce
import sys
import argparse

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--gpu_update_sleep_time", type=int, default=5)
    parser.add_argument("--scheduler_update_sleep_time", type=int, default=2)
    parser.add_argument("--waiting_time", type=int, default=20)
    parser.add_argument("--dataset_update_timeout", type=int, default=120)
    parser.add_argument("--start_load_job", type=bool, default=True)
    parser.add_argument("--start_load_dataset", type=bool, default=True)
    parser.add_argument("--finished_clear_job", type=bool, default=True)
    parser.add_argument("--finished_clear_dataset", type=bool, default=True)
    
    args = parser.parse_args()
    return args

class Dispatcher(object):
    def __init__(self, jobs_list, datasets_map):
        jobs_id_list = [x for x in range(len(jobs_list))]
        jobs_detail = list(map(lambda x: [x[0], x[1]], zip(jobs_id_list, jobs_list)))
        self.jobs_detail = jobs_detail
        self.finished_labels = {job_id:False for job_id, _ in self.jobs_detail}
        self.dispatch_jobs_count = 0

        self.datasets_map = datasets_map
        count = 0
        for sub_map in datasets_map.values():
            count += len(sub_map)
        self.all_datasets_count = count
        self.dispatch_datasets_count = 0
        
        self.all_finished = False

        self.all_start_time = time.time()
        self.current_time = 0

        current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
        file_log_name = 'schedule-review-%s' % (current_time)
        self.summary_writer_path = TENSORBOARD_PATH + "/{}".format(file_log_name)
        
    def dispatch_jobs(self, sched_ip, sched_port):
        def thread_func_timely_dispatch_job(sched_ip, sched_port):
            while not self.all_finished:
                count = self.dispatch_jobs_count
                dispatch_jobs_detail = []
                for index in range(len(self.jobs_detail)):
                    job_id, info = self.jobs_detail[index]
                    
                    need_submit_time = info["time"]
                    has_submited_flag = info["submited"]
                    if not has_submited_flag and need_submit_time >= self.current_time:
                        self.jobs_detail[index][1]["submited"] = True
                        count += 1
                        dispatch_jobs_detail.append([job_id, info])
                if count > self.dispatch_jobs_count:
                    self.dispatch_jobs_count = count
                    client = self.get_zerorpc_client(sched_ip, sched_port)
                    client.update_jobs(dispatch_jobs_detail)
                if self.dispatch_datasets_count == len(self.jobs_detail):
                    print("Finished Job Dispatch!")
                    break
                time.sleep(1)
            print("Thread [thread_func_timely_dispatch_job] finished!")
        p = threading.Thread(target=thread_func_timely_dispatch_job, args=(sched_ip, sched_port), daemon=True)
        p.start()
        return p

    def finished_job_callback(self, job_id):
        print("success: ", job_id)
        self.finished_labels[job_id] = True        

    def sched_update_dataset(self, sched_ip, sched_port, dataset_update_timeout):
        def thread_func_timely_dispatch_dataset(sched_ip, sched_port, dataset_update_timeout):
            sub_train_dataset_config_path = SUB_TRAIN_DATASET_CONFIG_PATH
            test_dataset_config_path = TEST_DATASET_CONFIG_PATH
            while not self.all_finished:
                count = self.dispatch_datasets_count
                dispatch_datasetidentifier_2_epsilon_capacity = {}
                for dataset_name in self.datasets_map:
                    for sub_train_dataset_identifier in self.datasets_map[dataset_name]:
                        need_submit_time = self.datasets_map[dataset_name][sub_train_dataset_identifier]["time"]
                        capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["capacity"]
                        has_submited_flag = self.datasets_map[dataset_name][sub_train_dataset_identifier]["submited"]
                        if not has_submited_flag and need_submit_time >= self.current_time:
                            self.datasets_map[dataset_name][sub_train_dataset_identifier]["submited"] = True
                            count += 1
                            if dataset_name not in dispatch_datasetidentifier_2_epsilon_capacity:
                                dispatch_datasetidentifier_2_epsilon_capacity[dataset_name] = {}
                            dispatch_datasetidentifier_2_epsilon_capacity[dataset_name][sub_train_dataset_identifier] = capacity
                if count > self.dispatch_datasets_count:
                    self.dispatch_datasets_count = count
                    client = self.get_zerorpc_client(sched_ip, sched_port, timeout=dataset_update_timeout)
                    client.update_dataset(dispatch_datasetidentifier_2_epsilon_capacity, sub_train_dataset_config_path, test_dataset_config_path)
                if self.dispatch_datasets_count == self.all_datasets_count:
                    print("Finished Dataset Dispatch!")
                    break
                time.sleep(1)
            print("Thread [thread_func_timely_dispatch_dataset] finished!")
        p = threading.Thread(target=thread_func_timely_dispatch_dataset, args=(sched_ip, sched_port, dataset_update_timeout), daemon=True)
        p.start()
        return p
        
    def sched_update_current_time(self):
        def thread_func_timely_update_time():
            while not self.all_finished:
                self.current_time = time.time() - self.all_start_time
                time.sleep(1)
            print("Thread [thread_func_timely_update_time] finished!")
        p = threading.Thread(target=thread_func_timely_update_time, daemon=True)
        p.start()
        return p

    

    def get_zerorpc_client(self, ip, port, timeout=30):
        tcp_ip_port = "tcp://{}:{}".format(ip, port)
        client = zerorpc.Client(timeout=timeout)
        client.connect(tcp_ip_port)
        return client

    def sched_clear_all_jobs(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.clear_all_jobs()

    def sched_clear_all_datasets(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.clear_all_datasets()

    def sched_dispatch_start(self, ip, port, scheduler_update_sleep_time):
        client = self.get_zerorpc_client(ip, port)
        client.sched_dispatch_start(scheduler_update_sleep_time, self.summary_writer_path)
    
    def sched_end(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.schd_end()

    def sched_update_gpu_status_start(self, ip, port,  init_gpuidentifiers, sleep_time):
        client = self.get_zerorpc_client(ip, port)
        client.sched_update_gpu_status_start(init_gpuidentifiers, sleep_time)

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

if __name__ == "__main__":
    args = get_df_config()

    sched_ip = SCHE_IP
    sched_port = SCHE_PORT
    dispatcher_ip = DISPATCHER_IP
    dispatcher_port = DISPATCHER_PORT
    
    init_gpuidentifiers = INIT_WORKERIDENTIFIERS
    gpu_update_sleep_time = args.gpu_update_sleep_time
    dataset_update_timeout = args.dataset_update_timeout
    scheduler_update_sleep_time = args.scheduler_update_sleep_time
    waiting_time = args.waiting_time
    
    # 需要增加时间标签!
    jobs_list = [
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 2, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 256.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 4, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 256.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 6, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 256.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 256.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 64.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 64.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 64.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 64.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 32.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 32.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 32.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 32.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 16.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 16.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 16.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    },
    {
        "time": 1,
        "submited": False,
        "target_func": "opacus_split_review",
        "model_name": "LSTM-split",
        "dataset_name": "Home_and_Kitchen",
        "label_type": "sentiment",
        "select_num": 8, # 2
        "early_stopping": False,
        "device": 0,
        "LR": 1e-3,
        "EPSILON": 16.0,
        "EPOCH_SET_EPSILON": False,
        "DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": 128,
        "MAX_PHYSICAL_BATCH_SIZE": 32,
        "EPOCHS": 20,
        "train_configs": {
            "n_layer": 2,
            "hidden_size": 40,
            "embedding_size": 100,
            "sequence_length": 50,
        },
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    }
    ]

    
    datasets_list = {
        "Home_and_Kitchen": {
            "train_sub_0": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_1": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_2": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_3": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_4": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_5": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_6": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_7": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_8": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_9": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_10": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_11": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_12": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_13": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_14": {
                "submited": False,
                "capacity": 50000.0,
                "time": 0,
            },
            "train_sub_15": {
                "submited": False,
                "capacity": 5000.0,
                "time": 0,
            }
        }
    }

    processes = []
    try:
        dispatcher = Dispatcher(jobs_list, datasets_list)
        remote_server_p = scheduler_listener_func(dispatcher, dispatcher_port)
        processes.append(remote_server_p)
        
        dispatcher.sched_update_gpu_status_start(sched_ip, sched_port, init_gpuidentifiers, gpu_update_sleep_time)
        # time.sleep(gpu_update_sleep_time)
        
        if args.start_load_dataset:
            dataset_p = dispatcher.sched_update_dataset(sched_ip, sched_port, dataset_update_timeout)
            processes.append(dataset_p)
        if args.start_load_job:
            job_p = dispatcher.dispatch_jobs(sched_ip, sched_port)
            processes.append(job_p)

        print("Waiting for load datasets and jobs {} s".format(waiting_time))
        time.sleep(waiting_time)
        dispatcher.sched_dispatch_start(sched_ip, sched_port, scheduler_update_sleep_time)

        time_p = dispatcher.sched_update_current_time()
        processes.append(time_p)
        
        # 主线程的最后一个操作!
        all_finished_label = reduce(lambda a, b: a and b, dispatcher.finished_labels.values())
        while not all_finished_label:
            time.sleep(gpu_update_sleep_time)
            all_finished_label = reduce(lambda a, b: a and b, dispatcher.finished_labels.values())
        print("logically all stoped!")
        dispatcher.all_finished = True
        dispatcher.sched_end(sched_ip, sched_port)
        if args.finished_clear_job:
            dispatcher.sched_clear_all_jobs(sched_ip, sched_port)
        if args.finished_clear_dataset:
            dispatcher.sched_clear_all_datasets(sched_ip, sched_port)
        
        print("Waiting for stop threads {} s".format(waiting_time))
        time.sleep(waiting_time)
        sys.exit(0)
    except Exception as e:
        print("[xlc] Exception: ", e)