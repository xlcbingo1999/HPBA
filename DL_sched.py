import zerorpc
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.data_loader import fetch_new_dataset
from utils.get_profiler_significance import get_profiler_selection_result
from utils.global_functions import FAILED_RESULT_KEY, JOB_STATUS_KEY, JOB_STATUS_UPDATE_PATH, DATASET_STATUS_KEY, add_2_map, normal_counter
from utils.global_variable import SCHE_IP, SCHE_PORT, INIT_WORKERIDENTIFIERS, INIT_WORKERIP_2_PORTS, GPU_PATH
from functools import reduce

import json
import time

def DL_server_do_jobs(job_id, origin_info, worker_ip, worker_port, worker_gpu_id, worker_dataset_config, summary_writer_path):
    client = zerorpc.Client()
    client.connect("tcp://{}:{}".format(worker_ip, worker_port))
    
    client.begin_job(job_id, worker_gpu_id, worker_dataset_config, origin_info, summary_writer_path)

class Scheduler_server(object):
    def __init__(self, sched_ip, sched_port, init_workerip_2_ports):
        self.sched_ip = sched_ip
        self.sched_port = sched_port
        self.workerip_2_ports = init_workerip_2_ports

        self.all_finished = False
        self.sched_thread = None
        self.gpu_thread = None
        
        self.gpuidentifier_2_gpu_status = {}
        self.gpuidentifier_2_gpu_metadata = {}
        
        self.sub_train_datasetidentifier_2_dataset_status = {} # 这里必须是一个可以伸缩的map
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.test_datasetname_2_metadata = {}
        
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SCHE: [], 
                                JOB_STATUS_KEY.DONE_GPU_SCHED: [], 
                                JOB_STATUS_KEY.DONE_DATASET_SCHED: [], 
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}
        self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}
        self.jobid_2_target_epsilon = {}
        self.jobid_2_real_epsilon = {}

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}

        

    def clear_all_jobs(self):
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SCHE: [], 
                                JOB_STATUS_KEY.DONE_GPU_SCHED: [], 
                                JOB_STATUS_KEY.DONE_DATASET_SCHED: [], 
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}
        self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}
        self.jobid_2_target_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}
        for worker_ip in self.workerip_2_ports:
            worker_port = self.workerip_2_ports[worker_ip]
            client = self.get_zerorpc_client(worker_ip, worker_port)
            client.clear_all_jobs()
        print("success clear all jobs")

    def clear_all_datasets(self):
        self.sub_train_datasetidentifier_2_dataset_status = {}
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.test_datasetname_2_metadata = {}
        print("success clear all datasets")

    def get_zerorpc_client(self, ip, port):
        tcp_ip_port = "tcp://{}:{}".format(ip, port)
        client = zerorpc.Client()
        client.connect(tcp_ip_port)
        return client

    def get_worker_identifier_detail(self, worker_identifier):
        worker_ip, worker_gpu_id = worker_identifier.split("-")
        return worker_ip, int(worker_gpu_id)

    def update_dataset(self, init_datasetidentifier_2_epsilon_capacity, sub_train_dataset_config_path, init_test_dataset_config_path):
        """
        外部调用
        init_datasetidentifiers(map): {dataset_name: {sub_identifier: {}}}
        """
        datset_configs = {}
        with open(sub_train_dataset_config_path, "r") as f:
            datset_configs = json.load(f)
        for init_dataset_name in init_datasetidentifier_2_epsilon_capacity:
            if init_dataset_name not in self.test_datasetname_2_metadata:
                with open(init_test_dataset_config_path, "r") as f:
                    test_datset_configs = json.load(f)
                self.test_datasetname_2_metadata[init_dataset_name] = test_datset_configs[init_dataset_name]
            
            dataset_identifier_2_capacity_map = init_datasetidentifier_2_epsilon_capacity[init_dataset_name] # 这里会得到一个map
            if init_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_exhausted_time[init_dataset_name] = {}
            
            for identifier in dataset_identifier_2_capacity_map:
                if identifier not in datset_configs[init_dataset_name]:
                    print("[warning] {} not in dataset config!".format(identifier))
                    continue
                if identifier in self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name]:
                    print("[warning] {} already in dataset config!".format(identifier))
                    continue
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name][identifier] = datset_configs[init_dataset_name][identifier]
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name][identifier] = time.time()
                print("sucess update dataset [{}-{}]".format(init_dataset_name, identifier))

        # self.report_status("success update dataset!")
        

    def update_gpu(self, init_gpuidentifiers):
        """
        这个函数不能传, 必须时刻读取共享文件系统中的数据情况, 慢操作, 开Thread读取
        """
        def read_gpu_state_from_file(gpu_identifier):
            gpu_config_path = GPU_PATH + "/{}.json".format(gpu_identifier)
            with open(gpu_config_path, "r") as f:
                try:
                    metadata = json.load(f)
                except Exception as e:
                    print("read {} exception: {}".format(gpu_config_path, e))
                    f.close()
                    return 
                self.gpuidentifier_2_gpu_metadata[gpu_identifier] = metadata
                if self.gpuidentifier_2_gpu_metadata[gpu_identifier]["free_mem"] > 0.0:
                    self.gpuidentifier_2_gpu_status[gpu_identifier] = True
                else:
                    self.gpuidentifier_2_gpu_status[gpu_identifier] = False
                f.close()

        for gpu_identifier in init_gpuidentifiers:
            threading.Thread(target=read_gpu_state_from_file, args=(gpu_identifier, ), daemon=True).start()

    def update_jobs(self, jobs_detail):
        for id, origin_info in jobs_detail:
            if id in self.jobid_2_status:
                print("Waring: job {} has existed!".format(id))
                continue
            else:
                self.jobid_2_status[id] = JOB_STATUS_KEY.NO_SCHE
                self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE].append(id)
                self.jobid_2_results[id] = None
                self.jobid_2_origininfo[id] = origin_info
                self.jobid_2_gputarget[id] = None
                self.jobid_2_datasettargetconfig[id] = {
                    "dataset_name": origin_info["dataset_name"],
                    "label_type": origin_info["label_type"]
                }
                self.jobid_2_trainconfig[id] = {}
                self.jobid_2_target_epsilon[id] = origin_info["EPSILON"]
                self.jobid_2_real_epsilon[id] = 0
                self.jobid_2_submited_time[id] = time.time()
                print("success add new job {}".format(id))
        # self.report_sched_status("after add all jobs")

    def worker_finished_job_callback(self, job_id, origin_info, result):
        print("=========  Scheduler: Job Finished! ===========")
        print("job_id: {}".format(job_id))
        print("origin_info: {}".format(origin_info))
        print("result: {}".format(result))
        print("====================")
        self.sche_update_job_status(job_id, JOB_STATUS_KEY.FINISHED)
        self.sche_reflash_job_status([job_id], JOB_STATUS_KEY.RUNNING, JOB_STATUS_KEY.FINISHED)
        self.jobid_2_finished_time[job_id] = time.time()
        dispatcher_ip = origin_info["dispatcher_ip"]
        dispatcher_port = origin_info["dispatcher_port"]
        dispatcher_client = self.get_zerorpc_client(dispatcher_ip, dispatcher_port)
        dispatcher_client.finished_job_callback(job_id)
        self.jobid_2_results[job_id] = result
        self.jobid_2_real_epsilon[job_id] = result["epsilon_consume"]
        remain_epsilon = self.jobid_2_target_epsilon[job_id] - self.jobid_2_real_epsilon[job_id]
        dataset_name = self.jobid_2_datasettargetconfig[job_id]["dataset_name"]
        datablock_identifiers = self.jobid_2_datasettargetconfig[job_id]["selected_datablock_identifiers"]
        for identifier in datablock_identifiers:
            self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] += remain_epsilon
            if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] > 0.0:
                self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED

    def worker_failed_job_callback(self, job_id, failed_result_key):
        print("=========  Scheduler: Job Failed! ===========")
        print("job_id: {}".format(job_id))
        print("failed_result_key: {}".format(failed_result_key))
        print("====================")

    def worker_gpu_status_callback(self, worker_identifier, new_status):
        # 应该是一个单点通信, 单个worker直接和调度器通信即可
        self.gpuidentifier_2_gpu_status[worker_identifier] = new_status
        print("Scheduler: Worker's gpu [{}] Status Update to {}".format(worker_identifier, new_status))

    def sche_update_job_status(self, job_id, new_status):
        self.jobid_2_status[job_id] = new_status

    def sche_reflash_job_status(self, job_ids, origin_status, new_status):
        for job_id in job_ids:
            self.status_2_jobid[origin_status].remove(job_id)
            self.status_2_jobid[new_status].append(job_id)

    def report_status(self, location):
        print("======== Scheduler Status in {} ========".format(location))
        print("self.jobid_2_status: {}".format(self.jobid_2_status))
        print("self.status_2_jobid: {}".format(self.status_2_jobid))
        print("self.jobid_2_gputarget: {}".format(self.jobid_2_gputarget))
        print("self.sub_train_datasetidentifier_2_dataset_status: ", self.sub_train_datasetidentifier_2_dataset_status)
        print("self.sub_train_datasetidentifier_2_dataset_metadata: ", self.sub_train_datasetidentifier_2_dataset_metadata)
        print("self.sub_train_datasetidentifier_2_epsilon_capacity: ", self.sub_train_datasetidentifier_2_epsilon_capacity)
        print("self.sub_train_datasetidentifier_2_epsilon_remain: ", self.sub_train_datasetidentifier_2_epsilon_remain)
        print("==================================")

    def get_target_job_status_update_path_and_status(self, job_id, operator):
        origin_status = self.jobid_2_status[job_id]
        update_path = None
        new_status = None
        if operator == "gpu":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                # print("in: gpu 1")
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED
                new_status = JOB_STATUS_KEY.DONE_GPU_SCHED
            elif origin_status == JOB_STATUS_KEY.DONE_DATASET_SCHED:
                # print("in: gpu 2")
                update_path = JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED 
                new_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif operator == "dataset":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                # print("in: dataset 1")
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED
                new_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
            elif origin_status == JOB_STATUS_KEY.DONE_GPU_SCHED:
                # print("in: dataset 2")
                update_path = JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED
                new_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        # print("origin_status: ", origin_status)
        # print("update_path: ", update_path)
        # print("new_status: ", new_status)
        return update_path, new_status

    def get_job_status_update_origin_target(self, status_update_path):
        if status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_GPU_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_GPU_SCHED
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        return origin_status, target_status

    def get_scheduling_datablock_result(self, target_select_dataset_name, target_epsilon_require, target_select_num):
        if target_select_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
            return [], [], [], {}
        train_all_label_distribution = {}
        sub_train_datasetidentifier_2_label_distribution = {}
        for datablock_identifier in self.sub_train_datasetidentifier_2_dataset_status[target_select_dataset_name].keys():
            train_all_label_distribution = add_2_map(self.sub_train_datasetidentifier_2_dataset_metadata[target_select_dataset_name][datablock_identifier]["label_distribution"], train_all_label_distribution)
            if self.sub_train_datasetidentifier_2_dataset_status[target_select_dataset_name][datablock_identifier] == DATASET_STATUS_KEY.SUBMITED and self.sub_train_datasetidentifier_2_epsilon_remain[target_select_dataset_name][datablock_identifier] > target_epsilon_require:
                sub_train_datasetidentifier_2_label_distribution[datablock_identifier] = self.sub_train_datasetidentifier_2_dataset_metadata[target_select_dataset_name][datablock_identifier]["label_distribution"] 
        selected_datablock_identifiers, not_selected_datablock_identifiers, \
            final_scores, selected_label_distribution = \
            get_profiler_selection_result(train_all_label_distribution, sub_train_datasetidentifier_2_label_distribution, target_select_num)
        return selected_datablock_identifiers, not_selected_datablock_identifiers, final_scores, selected_label_distribution

    def sched_dispatch_main(self, summary_writer_path):
        # 未调度先调度
        to_reflash_job_ids = {
            JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED: [],
            JOB_STATUS_UPDATE_PATH.GPUSCHED_2_ALLSCHED: []
        }
        to_sched_dataset_jobids = self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE] + self.status_2_jobid[JOB_STATUS_KEY.DONE_GPU_SCHED]
        for job_id in to_sched_dataset_jobids:
            dataset_name = self.jobid_2_datasettargetconfig[job_id]["dataset_name"]
            target_epsilon_require = self.jobid_2_target_epsilon[job_id]
            target_select_num = self.jobid_2_origininfo[job_id]["select_num"]
            
            # 需要使用复杂一点的调度策略了
            selected_datablock_identifiers, not_selected_datablock_identifiers, \
                final_scores, selected_label_distribution =\
                 self.get_scheduling_datablock_result(dataset_name, target_epsilon_require, target_select_num)
            if len(selected_datablock_identifiers) > 0:
                print("Job [{}] selected datablock identifiers: {}".format(job_id, selected_datablock_identifiers))
                self.jobid_2_datasettargetconfig[job_id]["selected_datablock_identifiers"] = selected_datablock_identifiers
                self.jobid_2_datasettargetconfig[job_id]["not_selected_datablock_identifiers"] = not_selected_datablock_identifiers
                self.jobid_2_datasettargetconfig[job_id]["train_dataset_raw_paths"] = [self.sub_train_datasetidentifier_2_dataset_metadata[dataset_name][identifier]["path"] for identifier in selected_datablock_identifiers]
                
                # print("[bug fix] self.test_datasetname_2_metadata[dataset_name]: ", self.test_datasetname_2_metadata[dataset_name])
                self.jobid_2_datasettargetconfig[job_id]["test_dataset_raw_path"] = self.test_datasetname_2_metadata[dataset_name]["path"]
                self.jobid_2_datasettargetconfig[job_id]["label_distributions"] = selected_label_distribution
                self.jobid_2_datasettargetconfig[job_id]["is_select"] = True
                
                consume_epsilon = self.jobid_2_origininfo[job_id]["EPSILON"]
                
                for identifier in selected_datablock_identifiers:
                    self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] -= consume_epsilon
                    if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] <= 0.0:
                        self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.EXHAUST
                        self.sub_train_datasetidentifier_2_exhausted_time[dataset_name][identifier] = time.time()

                status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "dataset")
                to_reflash_job_ids[status_update_path].append(job_id)

        for status_path in to_reflash_job_ids:
            origin_status, target_status = self.get_job_status_update_origin_target(status_path)
            for job_id in to_reflash_job_ids[status_path]:
                self.sche_update_job_status(job_id, target_status)
            self.sche_reflash_job_status(to_reflash_job_ids[status_path], origin_status, target_status)
        
        # self.report_status("after sched dataset to job")

        to_reflash_job_ids = {
            JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED: [],
            JOB_STATUS_UPDATE_PATH.DATASETSCHED_2_ALLSCHED: []
        }
        to_sched_gpu_jobids = self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE] + self.status_2_jobid[JOB_STATUS_KEY.DONE_DATASET_SCHED]
        for job_id in to_sched_gpu_jobids:
            all_workers_list = list(self.gpuidentifier_2_gpu_status.keys())
            if len(all_workers_list) <= 0:
                continue
            target_worker_id = job_id % len(all_workers_list) # 决定worker_gpu
            target_worker_identifier = all_workers_list[target_worker_id]
            
            if self.gpuidentifier_2_gpu_status[target_worker_identifier]: # 判断GPU的状态
                self.jobid_2_gputarget[job_id] = target_worker_identifier
                status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "gpu")
                to_reflash_job_ids[status_update_path].append(job_id)
            
        for status_path in to_reflash_job_ids:
            origin_status, target_status = self.get_job_status_update_origin_target(status_path)
            for job_id in to_reflash_job_ids[status_path]:
                self.sche_update_job_status(job_id, target_status)
            self.sche_reflash_job_status(to_reflash_job_ids[status_path], origin_status, target_status)
        
        self.placement_dispatch(summary_writer_path)

    def placement_dispatch(self, summary_writer_path):
        # 放置任务
        args = []
        to_reflash_job_ids = []
        
        for job_id in self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]:
            origin_info = self.jobid_2_origininfo[job_id]
            worker_dataset_config = self.jobid_2_datasettargetconfig[job_id]
            worker_identifier = self.jobid_2_gputarget[job_id]
            worker_ip, worker_gpu_id = self.get_worker_identifier_detail(worker_identifier)
            worker_port = self.workerip_2_ports[worker_ip]
            args.append([job_id, origin_info, worker_ip, worker_port, worker_gpu_id, worker_dataset_config, summary_writer_path])
            self.sche_update_job_status(job_id, JOB_STATUS_KEY.RUNNING)
            self.jobid_2_started_time[job_id] = time.time()
            to_reflash_job_ids.append(job_id)
        self.sche_reflash_job_status(to_reflash_job_ids, JOB_STATUS_KEY.DONE_ALL_SCHED, JOB_STATUS_KEY.RUNNING)
        if len(args) > 0:
            # self.report_status("after placement_dispatch all jobs")
            # 转置
            final_args = [[row[i] for row in args] for i in range(len(args[0]))]
            print("after placement_dispatch all job ids: [{}]".format(final_args[0]))
            with ThreadPoolExecutor(max_workers=len(args)) as pool:
                pool.map(DL_server_do_jobs, *final_args)

    def sched_dispatch_start(self, scheduler_update_sleep_time, summary_writer_path):
        def thread_func_timely_schedule(scheduler_update_sleep_time, summary_writer_path):
            time.sleep(scheduler_update_sleep_time)
            while not self.all_finished:
                self.sched_dispatch_main(summary_writer_path)
                time.sleep(scheduler_update_sleep_time)
            print("Thread [thread_func_timely_schedule] finished!")
        self.all_finished = False
        p = threading.Thread(target=thread_func_timely_schedule, args=(scheduler_update_sleep_time, summary_writer_path), daemon=True)
        self.sched_thread = p
        p.start()
        print("Thread [thread_func_timely_schedule] started!")
    
    def schd_end(self):
        self.all_finished = True
        self.sched_thread = None
        self.gpu_thread = None

    def sched_update_gpu_status_start(self, init_gpuidentifiers, sleep_time):
        def thread_func_timely_update_gpu(init_gpuidentifiers):
            while not self.all_finished:
                self.update_gpu(init_gpuidentifiers)
                time.sleep(sleep_time)
            print("Thread [thread_func_timely_update_gpu] finished!")
            
        p = threading.Thread(target=thread_func_timely_update_gpu, args=(init_gpuidentifiers, ), daemon=True)
        self.gpu_thread = p
        p.start()
        print("Thread [thread_func_timely_update_gpu] started!")
        
def scheduler_listener_func(scheduler_server_item):
    s = zerorpc.Server(scheduler_server_item)
    ip_port = "tcp://0.0.0.0:{}".format(scheduler_server_item.sched_port)
    s.bind(ip_port)
    print("DL_server running in {}".format(ip_port))
    s.run()
    print("print sth...")

if __name__ == "__main__":
    sched_ip = SCHE_IP
    sched_port = SCHE_PORT
    init_workeridentifiers = INIT_WORKERIDENTIFIERS
    init_workerip_2_ports = INIT_WORKERIP_2_PORTS

    scheduler_server_item = Scheduler_server(sched_ip, sched_port, init_workerip_2_ports)
    scheduler_listener_func(scheduler_server_item)