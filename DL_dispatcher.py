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
    parser.add_argument("--cal_significance_sleep_time", type=int, default=2)
    parser.add_argument("--placement_sleep_time", type=int, default=2)

    parser.add_argument("--waiting_time", type=int, default=10)
    parser.add_argument("--dataset_update_timeout", type=int, default=120)
    parser.add_argument("--start_load_job", action="store_true")
    parser.add_argument("--start_load_history_job", action="store_true")
    parser.add_argument("--start_load_dataset", action="store_true")
    parser.add_argument("--finished_clear_job", action="store_true")
    parser.add_argument("--finished_clear_dataset", action="store_true")

    parser.add_argument("--assignment_policy", type=str, default="HISPolicy")
    parser.add_argument("--significance_policy", type=str, default="HISOTDDPolicy")

    parser.add_argument('--pbg_comparison_cost_epsilons', type=float, nargs="+", default=0.0)
    parser.add_argument('--pbg_comparison_z_thresholds', type=float, nargs="+", default=0.7)
    parser.add_argument('--pbg_Ls', type=float, nargs="+", default=0.01)
    parser.add_argument('--pbg_Us', type=float, nargs="+", default=10.0)

    parser.add_argument('--his_betas', type=float, nargs="+", default=0.0)

    parser.add_argument('--dpf_his_betas', type=float, nargs="+", default=0.01)
    parser.add_argument('--dpf_his_waiting_queue_capacitys', type=int, nargs="+", default=10)
    
    args = parser.parse_args()
    return args

class Dispatcher(object):
    def __init__(self, jobs_list, history_jobs_list, datasets_map):
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
                dispatch_jobs_detail = {}
                for index in range(len(self.jobs_detail)):
                    job_id, info = self.jobs_detail[index]
                    
                    need_submit_time = info["time"]
                    has_submited_flag = info["submited"]
                    if not has_submited_flag and need_submit_time >= self.current_time:
                        self.jobs_detail[index][1]["submited"] = True
                        count += 1
                        dispatch_jobs_detail[job_id] = info
                if count > self.dispatch_jobs_count:
                    self.dispatch_jobs_count = count
                    client = self.get_zerorpc_client(sched_ip, sched_port)
                    client.update_jobs(dispatch_jobs_detail) # 提交上去后, 任务即进入NO_SCHED状态, 之后就是调度器自身会启动一个不断循环的获取计算Siginificane策略和调度策略
                if self.dispatch_datasets_count == len(self.jobs_detail):
                    print("Finished Job Dispatch!")
                    break
                time.sleep(1)
            print("Thread [thread_func_timely_dispatch_job] finished!")
        p = threading.Thread(target=thread_func_timely_dispatch_job, args=(sched_ip, sched_port), daemon=True)
        p.start()
        return p

    def dispatch_history_jobs(self, sched_ip, sched_port):
        client = self.get_zerorpc_client(sched_ip, sched_port)
        client.update_history_jobs(self.history_jobs_detail)

    
    def finished_job_callback(self, job_id):
        print("success: ", job_id)
        self.finished_labels[job_id] = True   

    def sched_update_dataset(self, sched_ip, sched_port, dataset_update_timeout):
        def thread_func_timely_dispatch_dataset(sched_ip, sched_port, dataset_update_timeout):
            while not self.all_finished:
                count = self.dispatch_datasets_count
                subtrain_datasetidentifier_info = {}
                for dataset_name in self.datasets_map:
                    for sub_train_dataset_identifier in self.datasets_map[dataset_name]:
                        need_submit_time = self.datasets_map[dataset_name][sub_train_dataset_identifier]["time"]
                        epsilon_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["epsilon_capacity"]
                        delta_capacity = self.datasets_map[dataset_name][sub_train_dataset_identifier]["delta_capacity"]
                        has_submited_flag = self.datasets_map[dataset_name][sub_train_dataset_identifier]["submited"]
                        if not has_submited_flag and need_submit_time >= self.current_time:
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
                    client = self.get_zerorpc_client(sched_ip, sched_port, timeout=dataset_update_timeout)
                    client.update_dataset(subtrain_datasetidentifier_info)
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

    def sched_dispatch_start(self, ip, port, cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_time):
        client = self.get_zerorpc_client(ip, port)
        client.cal_significance_dispatch_start(cal_significance_sleep_time)
        client.sched_dispatch_start(scheduler_update_sleep_time)
        client.placement_dispatch_start(placement_sleep_time, self.summary_writer_path)
    
    def sched_end(self, ip, port):
        client = self.get_zerorpc_client(ip, port)
        client.schd_end()

    def sched_update_gpu_status_start(self, ip, port,  init_gpuidentifiers, sleep_time):
        client = self.get_zerorpc_client(ip, port)
        client.sched_update_gpu_status_start(init_gpuidentifiers, sleep_time)

    def sched_init_sched_policy(self, ip, port, assignment_policy, significance_policy):
        client = self.get_zerorpc_client(ip, port)
        if assignment_policy == "PBGPolicy":
            comparison_cost_epsilon_list = args.pbg_comparison_cost_epsilons
            comparison_z_threshold_list = args.pbg_comparison_z_thresholds
            L_list = args.pbg_Ls
            U_list = args.pbg_Us
            assignment_args = (comparison_cost_epsilon_list, comparison_z_threshold_list, L_list, U_list)
        elif assignment_policy == "HISPolicy":
            beta_list = args.his_betas
            assignment_args = beta_list
        elif assignment_policy == "DPFHISPolicy":
            beta_list = args.dpf_his_betas
            waiting_queue_capacity_list = args.dpf_his_waiting_queue_capacitys
            assignment_args = (beta_list, waiting_queue_capacity_list)
        else:
            assignment_args = None
        client.sched_update_assignment_policy(assignment_policy, assignment_args)
        client.sched_update_significance_policy(significance_policy)

    def sched_init_history_policy(self, ip, port, history_jobs_map):
        client = self.get_zerorpc_client(ip, port)
        client.update_history_jobs(history_jobs_map)

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
    cal_significance_sleep_time = args.cal_significance_sleep_time
    placement_sleep_time = args.placement_sleep_time

    waiting_time = args.waiting_time

    jobs_list = [
        {
            "time": 1,
            "submited": False,
            "model_name": "CNN",
            "train_dataset_name": "EMNIST",
            "test_dataset_name": "EMNIST_MNIST-1000_1000", # "EMNIST_iid", "SVHN"
            "sub_test_key_ids": ["test_sub_0"],
            "datablock_select_num": 1,
            "LR": 1e-3,
            "EPSILON": 0.05,
            "DELTA": 1e-5,
            "MAX_GRAD_NORM": 1.2,
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "EPOCHS": 1,
            "priority_weight": 1.0,
            "dispatcher_ip": dispatcher_ip,
            "dispatcher_port": dispatcher_port,
        }
    ]
    history_jobs_list = [
        {
            "time": -1,
            "model_name": "CNN",
            "train_dataset_name": "EMNIST",
            "test_dataset_name": "EMNIST_MNIST-1000_1000", # "EMNIST_iid", "SVHN"
            "sub_test_key_ids": ["test_sub_0"],
            "datablock_select_num": 1,
            "LR": 1e-3,
            "EPSILON": 0.05,
            "DELTA": 1e-5,
            "MAX_GRAD_NORM": 1.2,
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "EPOCHS": 1,
            "priority_weight": 1.0,
            "dispatcher_ip": dispatcher_ip,
            "dispatcher_port": dispatcher_port,
        }, {
            "time": -2,
            "model_name": "CNN",
            "train_dataset_name": "EMNIST",
            "test_dataset_name": "EMNIST-2000", # "EMNIST_iid", "SVHN"
            "sub_test_key_ids": ["test_sub_0"],
            "datablock_select_num": 1,
            "LR": 1e-3,
            "EPSILON": 0.05,
            "DELTA": 1e-5,
            "MAX_GRAD_NORM": 1.2,
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "EPOCHS": 1,
            "priority_weight": 1.0,
            "dispatcher_ip": dispatcher_ip,
            "dispatcher_port": dispatcher_port,
        }, {
            "time": -3,
            "model_name": "CNN",
            "train_dataset_name": "EMNIST",
            "test_dataset_name": "MNIST-2000", # "EMNIST_iid", "SVHN"
            "sub_test_key_ids": ["test_sub_0"],
            "datablock_select_num": 1,
            "LR": 1e-3,
            "EPSILON": 0.05,
            "DELTA": 1e-5,
            "MAX_GRAD_NORM": 1.2,
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "EPOCHS": 1,
            "priority_weight": 1.0,
            "dispatcher_ip": dispatcher_ip,
            "dispatcher_port": dispatcher_port,
        }
        
    ]
    datasets_list = {
        "EMNIST": {
            "train_sub_0": {
                "submited": False,
                "epsilon_capacity": 5.0,
                "delta_capacity": 1e-7,
                "time": 0,
            },
            "train_sub_1": {
                "submited": False,
                "epsilon_capacity": 5.0,
                "delta_capacity": 1e-7,
                "time": 0,
            },
            "train_sub_2": {
                "submited": False,
                "epsilon_capacity": 5.0,
                "delta_capacity": 1e-7,
                "time": 0,
            },
            "train_sub_3": {
                "submited": False,
                "epsilon_capacity": 5.0,
                "delta_capacity": 1e-7,
                "time": 0,
            },
            "train_sub_4": {
                "submited": False,
                "epsilon_capacity": 5.0,
                "delta_capacity": 1e-7,
                "time": 0,
            },
            "train_sub_5": {
                "submited": False,
                "epsilon_capacity": 5.0,
                "delta_capacity": 1e-7,
                "time": 0,
            },
        }
    } # 注意: 这里时间全部设置为0即可
    processes = []
    try:
        dispatcher = Dispatcher(jobs_list, history_jobs_list, datasets_list)
        remote_server_p = scheduler_listener_func(dispatcher, dispatcher_port)
        processes.append(remote_server_p)

        dispatcher.sched_init_sched_policy(sched_ip, sched_port, args.assignment_policy, args.significance_policy)
        dispatcher.sched_update_gpu_status_start(sched_ip, sched_port, init_gpuidentifiers, gpu_update_sleep_time)
                
        if args.start_load_dataset:
            dataset_p = dispatcher.sched_update_dataset(sched_ip, sched_port, dataset_update_timeout)
            processes.append(dataset_p)
        if args.start_load_history_job:
            history_job_p = dispatcher.dispatch_history_jobs(sched_ip, sched_port)
        if args.start_load_job:
            job_p = dispatcher.dispatch_jobs(sched_ip, sched_port)
            processes.append(job_p)

        print("Waiting for load datasets and jobs {} s".format(waiting_time))
        time.sleep(waiting_time)
        dispatcher.sched_dispatch_start(sched_ip, sched_port, cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_time)

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