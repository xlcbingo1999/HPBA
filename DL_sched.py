import zerorpc
from concurrent.futures import ThreadPoolExecutor
import threading

import copy
import numpy as np
import random
from utils.data_loader import fetch_new_dataset
from utils.get_profiler_significance import get_profiler_selection_result
from utils.global_functions import FAILED_RESULT_KEY, JOB_STATUS_KEY, JOB_STATUS_UPDATE_PATH, DATASET_STATUS_KEY, add_2_map, normal_counter
from utils.global_variable import SCHE_IP, SCHE_PORT, INIT_WORKERIDENTIFIERS, INIT_WORKERIP_2_PORTS, GPU_PATH, LOGGING_PATH
from utils.logging_tools import get_logger

from policies.PBG import PBGPolicy
from policies.Sage import SagePolicy
from policies.HIS import HISPolicy
from policies.DPF_HIS_event import DPFHISPolicy
from policies.Offline import OfflinePolicy
from significance_policies.HISOTDD import HISOTDDPolicy
from significance_policies.Temp import TempPolicy


from functools import reduce

import json
import time

def DL_server_do_jobs(job_id, origin_info, update_sched_epoch_num, worker_ip, worker_port, worker_gpu_id, worker_dataset_config, summary_writer_path, logging_file_path):
    client = zerorpc.Client()
    client.connect("tcp://{}:{}".format(worker_ip, worker_port))
    
    client.begin_job(job_id, worker_gpu_id, worker_dataset_config, origin_info, update_sched_epoch_num, summary_writer_path, logging_file_path)

class Scheduler_server(object):
    def __init__(self, sched_ip, sched_port, init_workerip_2_ports):
        self.sched_ip = sched_ip
        self.sched_port = sched_port
        self.workerip_2_ports = init_workerip_2_ports

        self.all_finished = False
        self.sched_thread = None
        self.cal_significance_thread = None
        self.placement_thread = None
        self.gpu_thread = None

        self.update_sched_epoch_num = 1
        
        self.gpuidentifier_2_gpu_status = {}
        self.gpuidentifier_2_gpu_metadata = {}
        self.gpuidentifier_2_jobinstance_oneshot = {}
        
        self.sub_train_datasetidentifier_2_dataset_status = {} # 这里必须是一个可以伸缩的map
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.sub_train_datasetidentifier_2_train_type = {}
        self.test_datasetname_2_metadata = {}
        
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SCHE: [], 
                                JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL: [],
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        
        self.finished_update_init_history_jobs = False
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}

        # self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

        self.jobid_2_target_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_priority_weight = {}

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}
        self.jobid_2_train_dataset_name = {}
        self.jobid_2_sub_train_key_ids = {}
        self.jobid_2_datablock_selected_num = {}
        self.jobid_2_test_dataset_name = {}
        self.jobid_2_sub_test_key_ids = {}
        self.jobid_2_significance = {}
        self.jobid_2_arrival_index = {}

        self.jobid_2_max_epochs = {}
        self.jobid_2_real_sched_epochs = {}
        self.jobid_2_failed_epochs = {}
        self.jobid_2_current_epochs = {}

        self.job_sequence_all_num = 0
        self.history_job_priority_weights = []
        self.history_job_budget_consumes = []
        self.history_job_train_dataset_name = []
        self.history_job_target_selected_num = []
        self.history_job_test_dataset_name = []
        self.history_job_sub_test_key_ids = []
        self.history_job_significance = []

        self.assignment_policy = None
        self.significance_policy = None

        self.initialize_seeds(1234)

        current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
        file_log_name = 'schedule-review-%s' % (current_time)
        logger_path = '%s/%s.log' % (LOGGING_PATH, file_log_name)
        self.logger = get_logger(logger_path, enable_multiprocess=False)

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

        self.job_generator = random.Random()
        self.job_generator.seed(seed+2)

    def check_all_finished_or_failed(self):
        return (len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]) <= 0 
            and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.RUNNING]) <= 0
        )

    def clear_all_jobs(self):
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {JOB_STATUS_KEY.NO_SCHE: [], 
                                JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL: [],
                                JOB_STATUS_KEY.DONE_ALL_SCHED: [],
                                JOB_STATUS_KEY.RUNNING: [], 
                                JOB_STATUS_KEY.FINISHED: [],
                                JOB_STATUS_KEY.FAILED: []}
        self.finished_update_init_history_jobs = False
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}
        
        # self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

        self.jobid_2_target_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_priority_weight = {}
        self.jobid_2_train_dataset_name = {}
        self.jobid_2_sub_train_key_ids = {}
        self.jobid_2_datablock_selected_num = {}
        self.jobid_2_test_dataset_name = {}
        self.jobid_2_sub_test_key_ids = {}
        self.jobid_2_significance = {}
        self.jobid_2_arrival_index = {}

        self.jobid_2_max_epochs = {}
        self.jobid_2_real_sched_epochs = {}
        self.jobid_2_failed_epochs = {}
        self.jobid_2_current_epochs = {}

        self.jobid_2_submited_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}

        self.job_sequence_all_num = 0
        self.history_job_priority_weights = []
        self.history_job_budget_consumes = []
        self.history_job_train_dataset_name = []
        self.history_job_target_selected_num = []
        self.history_job_test_dataset_name = []
        self.history_job_sub_test_key_ids = []
        self.history_job_significance = []

        print("success clear all jobs")

    def clear_all_datasets(self):
        self.sub_train_datasetidentifier_2_dataset_status = {}
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_submited_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.sub_train_datasetidentifier_2_train_type = {}
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

    def update_dataset(self, init_subtrain_datasets_map):
        dispatch_datasetidentifier_2_epsilon_capacity = {}
        for dataset_name in init_subtrain_datasets_map:
            for sub_train_dataset_identifier in init_subtrain_datasets_map[dataset_name]:
                capacity = init_subtrain_datasets_map[dataset_name][sub_train_dataset_identifier]["epsilon_capacity"]
                if dataset_name not in dispatch_datasetidentifier_2_epsilon_capacity:
                    dispatch_datasetidentifier_2_epsilon_capacity[dataset_name] = {}
                dispatch_datasetidentifier_2_epsilon_capacity[dataset_name][sub_train_dataset_identifier] = capacity

        for init_dataset_name in dispatch_datasetidentifier_2_epsilon_capacity:
            dataset_identifier_2_capacity_map = dispatch_datasetidentifier_2_epsilon_capacity[init_dataset_name] # 这里会得到一个map
            print("success add datset[{}] with {} blocks".format(init_dataset_name, len(dataset_identifier_2_capacity_map)))
            if init_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_exhausted_time[init_dataset_name] = {}
            
            for identifier in dataset_identifier_2_capacity_map:
                if identifier not in init_subtrain_datasets_map[init_dataset_name]:
                    print("[warning] {} not in dataset config!".format(identifier))
                    continue
                if identifier in self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name]:
                    print("[warning] {} already in dataset config!".format(identifier))
                    continue
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name][identifier] = init_subtrain_datasets_map[init_dataset_name][identifier]
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_submited_time[init_dataset_name][identifier] = time.time()

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
                if gpu_identifier not in self.gpuidentifier_2_jobinstance_oneshot:
                    self.gpuidentifier_2_jobinstance_oneshot[gpu_identifier] = None
                self.gpuidentifier_2_gpu_metadata[gpu_identifier] = metadata
                if self.gpuidentifier_2_gpu_metadata[gpu_identifier]["free_mem"] > 0.0:
                    self.gpuidentifier_2_gpu_status[gpu_identifier] = True
                else:
                    self.gpuidentifier_2_gpu_status[gpu_identifier] = False
                f.close()

        for gpu_identifier in init_gpuidentifiers:
            threading.Thread(target=read_gpu_state_from_file, args=(gpu_identifier, ), daemon=True).start()
            
    def update_jobs(self, jobs_detail_map): # 每次可以增加一批任务
        count = 0
        for id in jobs_detail_map:
            origin_info = jobs_detail_map[id]
            if id in self.jobid_2_status:
                print("Waring: job {} has existed!".format(id))
                continue
            else:
                self.jobid_2_status[id] = JOB_STATUS_KEY.NO_SCHE
                self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE].append(id)
                self.jobid_2_results[id] = None
                self.jobid_2_origininfo[id] = origin_info
                self.jobid_2_gputarget[id] = None
                # self.jobid_2_datasettargetconfig[id] = {}
                self.jobid_2_trainconfig[id] = {}
                target_epsilon_consume = origin_info["EPSILON"]
                self.jobid_2_target_epsilon[id] = target_epsilon_consume
                self.jobid_2_real_epsilon[id] = 0
                self.jobid_2_submited_time[id] = origin_info["time"]
                self.jobid_2_priority_weight[id] = origin_info["priority_weight"]
                train_dataset_name = origin_info["train_dataset_name"]
                self.jobid_2_train_dataset_name[id] = train_dataset_name
                self.jobid_2_datablock_selected_num[id] = origin_info["datablock_select_num"]
                test_dataset_name = origin_info["test_dataset_name"]
                sub_test_key_ids = origin_info["sub_test_key_ids"]
                self.jobid_2_test_dataset_name[id] = test_dataset_name
                self.jobid_2_sub_test_key_ids[id] = sub_test_key_ids
                self.jobid_2_arrival_index[id] = count

                self.jobid_2_max_epochs[id] = origin_info["MAX_EPOCHS"]
                self.jobid_2_current_epochs[id] = 0
                self.jobid_2_real_sched_epochs[id] = 0
                self.jobid_2_failed_epochs[id] = 0
                
                count += 1
        self.job_sequence_all_num = count
        print("success add new jobs number: {}".format(self.job_sequence_all_num))

    def update_history_jobs(self, history_jobs_map):
        for id in sorted(history_jobs_map):
            self.history_job_priority_weights.append(history_jobs_map[id]["priority_weight"])
            target_epsilon_consume = history_jobs_map[id]["EPSILON"]
            self.history_job_budget_consumes.append(target_epsilon_consume)
            train_dataset_name = history_jobs_map[id]["train_dataset_name"]
            self.history_job_train_dataset_name.append(train_dataset_name)
            target_selected_num = history_jobs_map[id]["datablock_select_num"]
            self.history_job_target_selected_num.append(target_selected_num)
            test_dataset_name = history_jobs_map[id]["test_dataset_name"]
            self.history_job_test_dataset_name.append(test_dataset_name)
            sub_test_key_ids = history_jobs_map[id]["sub_test_key_ids"]
            self.history_job_sub_test_key_ids.append(sub_test_key_ids)

            if train_dataset_name in self.sub_train_datasetidentifier_2_dataset_status:
                all_significance_state = []
                for sub_train_key_id in self.sub_train_datasetidentifier_2_dataset_status[train_dataset_name]:
                    significance_state = {
                        "test_dataset_name": test_dataset_name,
                        "sub_test_key_ids": sub_test_key_ids,
                        "train_dataset_name": train_dataset_name,
                        "sub_train_key_ids": [sub_train_key_id]
                    }
                    all_significance_state.append(significance_state)
                result_d_map = self.get_job_datablock_significance_sync(all_significance_state)
                self.history_job_significance.append(result_d_map)
        print("success add new history jobs number: {}".format(len(history_jobs_map)))
        self.finished_update_init_history_jobs = True

    def sche_timely_update_history_job(self, priority_weight, EPSILON, train_dataset_name, datablock_selected_num, test_dataset_name, sub_test_key_ids, significance):
        self.history_job_priority_weights.append(priority_weight)
        self.history_job_budget_consumes.append(EPSILON)
        self.history_job_train_dataset_name.append(train_dataset_name)
        self.history_job_target_selected_num.append(datablock_selected_num)
        self.history_job_test_dataset_name.append(test_dataset_name)
        self.history_job_sub_test_key_ids.append(sub_test_key_ids)
        self.history_job_significance.append(significance)
        print("success add a new history job")

    def sche_reflash_job_status(self, job_id, origin_status, new_status):
        self.jobid_2_status[job_id] = new_status
        self.status_2_jobid[origin_status].remove(job_id)
        self.status_2_jobid[new_status].append(job_id)

    def job_add_to_history(self, job_id):
        self.sche_timely_update_history_job(self.jobid_2_priority_weight[job_id], self.jobid_2_target_epsilon[job_id],
                                            self.jobid_2_train_dataset_name[job_id], self.jobid_2_datablock_selected_num[job_id],
                                            self.jobid_2_test_dataset_name[job_id], self.jobid_2_sub_test_key_ids[job_id], self.jobid_2_significance[job_id])
    
    def worker_failed_job_callback(self, job_id, failed_result_key):
        print("=========  Scheduler: Job Failed! ===========")
        print("job_id: {}".format(job_id))
        print("failed_result_key: {}".format(failed_result_key))

        gpu_identifier = self.jobid_2_gputarget[job_id]
        self.jobid_2_gputarget[job_id] = None
        self.gpuidentifier_2_jobinstance_oneshot[gpu_identifier] = None

        print("====================")

    def report_status(self, location):
        print("======== Scheduler Status in {} ========".format(location))
        print("self.status_2_jobid")
        for status in self.status_2_jobid:
            print("status_2_jobid[{}]: {}".format(status, self.status_2_jobid[status]))
        # print("self.jobid_2_results")
        all_significance = 0.0
        all_accuracy = 0.0
        for job_id in self.jobid_2_results:
            if self.jobid_2_results[job_id] is not None:
                # print("jobid_2_results[{}]: {}".format(job_id, self.jobid_2_results[job_id]))
                all_significance += self.jobid_2_results[job_id]["significance"]
                all_accuracy += self.jobid_2_results[job_id]["accuracy"]
        print("self.sub_train_datasetidentifier_2_epsilon_remain")
        for datasetname in self.sub_train_datasetidentifier_2_epsilon_remain:
            for datasetidentifier in self.sub_train_datasetidentifier_2_epsilon_remain[datasetname]:
                print("sub_train_datasetidentifier_2_epsilon_remain[{}][{}]: {}".format(datasetname, datasetidentifier, self.sub_train_datasetidentifier_2_epsilon_remain[datasetname][datasetidentifier]))
        
        print("Finished Job num: {}".format(len(self.status_2_jobid[JOB_STATUS_KEY.FINISHED])))
        print("Failed Job num: {}".format(len(self.status_2_jobid[JOB_STATUS_KEY.FAILED])))
        print("all_significance: {}".format(all_significance))
        print("all_accuracy: {}".format(all_accuracy))
        print("==================================")

    def get_runtime_state(self, policy, job_id_2_train_dataset_name, job_id_2_target_epsilon_require, 
                        job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                        job_id_2_test_dataset_name, job_id_2_sub_test_key_ids, job_id_2_significance, job_id_2_arrival_index):
        state = {}
        state["job_id_2_train_dataset_name"] = job_id_2_train_dataset_name
        state["job_id_2_target_epsilon_require"] = job_id_2_target_epsilon_require
        state["job_id_2_target_datablock_selected_num"] = job_id_2_target_datablock_selected_num
        state["job_id_2_job_priority_weight"] = job_id_2_job_priority_weight
        state["job_id_2_test_dataset_name"] = job_id_2_test_dataset_name
        state["job_id_2_sub_test_key_ids"] = job_id_2_sub_test_key_ids
        state["job_id_2_significance"] = job_id_2_significance
        state["job_id_2_arrival_index"] = job_id_2_arrival_index

        state["current_sub_train_datasetidentifier_2_epsilon_remain"] = copy.deepcopy(self.sub_train_datasetidentifier_2_epsilon_remain)
        state["current_sub_train_datasetidentifier_2_epsilon_capcity"] = copy.deepcopy(self.sub_train_datasetidentifier_2_epsilon_capacity)

        if policy.name == "HISPolicy" or policy.name == "DPFHISPolicy":
            state["all_job_sequence_num"] = self.job_sequence_all_num
            state["history_job_priority_weights"] = self.history_job_priority_weights
            state["history_job_budget_consumes"] = self.history_job_budget_consumes
            state["history_job_train_dataset_name"] = self.history_job_train_dataset_name
            state["history_job_target_datablock_selected_num"] = self.history_job_target_selected_num
            state["history_job_significance"] = self.history_job_significance
 
        return state
    
    def get_significance_state(self, policy, train_dataset_name, datablock_identifier, test_type, target_epsilon_consume):
        signficance_state = {}
        signficance_state["train_dataset_name"] = train_dataset_name
        signficance_state["datablock_identifier"] = datablock_identifier
        signficance_state["test_type"] = test_type
        if policy.name == "TempPolicy":
            signficance_state["epsilon_consume"] = target_epsilon_consume
        return signficance_state
    
    def get_scheduling_datablock_result(self, policy, job_id_2_dataset_name, job_id_2_target_epsilon_require, 
                                        job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                                        job_id_2_test_dataset_name, job_id_2_sub_test_key_ids, 
                                        job_id_2_significance, job_id_2_arrival_index):
        job_2_selected_datablock_identifiers = []
        # 在这里接入算法?
        state = self.get_runtime_state(policy, job_id_2_dataset_name, job_id_2_target_epsilon_require, 
                                    job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                                    job_id_2_test_dataset_name, job_id_2_sub_test_key_ids, 
                                    job_id_2_significance, job_id_2_arrival_index)
        job_2_selected_datablock_identifiers, calcu_compare_epsilon = policy.get_allocation(state)
        # not_selected_datablock_identifiers = [tu[0] for tu in sub_train_sort[target_datablock_select_num:]]
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon


    def worker_finished_job_callback(self, job_id, origin_info, result):
        print("=========  Scheduler: Job Finished! ===========")
        print("job_id: {}".format(job_id))
        print("origin_info: {}".format(origin_info))
        print("result: {}".format(result))
        print("====================")
        self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.RUNNING, JOB_STATUS_KEY.FINISHED)
        self.jobid_2_finished_time[job_id] = time.time()
        dispatcher_ip = origin_info["dispatcher_ip"]
        dispatcher_port = origin_info["dispatcher_port"]
        dispatcher_client = self.get_zerorpc_client(dispatcher_ip, dispatcher_port)
        dispatcher_client.finished_job_callback(job_id)

        gpu_identifier = self.jobid_2_gputarget[job_id]
        self.jobid_2_gputarget[job_id] = None
        self.gpuidentifier_2_jobinstance_oneshot[gpu_identifier] = None

        self.jobid_2_results[job_id] = result
        self.jobid_2_real_epsilon[job_id] = result["epsilon_consume"]
        remain_epsilon = self.jobid_2_target_epsilon[job_id] - self.jobid_2_real_epsilon[job_id]
        train_dataset_name = self.jobid_2_train_dataset_name[job_id]
        datablock_identifiers = self.jobid_2_sub_train_key_ids[job_id]
        for identifier in datablock_identifiers:
            self.sub_train_datasetidentifier_2_epsilon_remain[train_dataset_name][identifier] += remain_epsilon
            if self.sub_train_datasetidentifier_2_epsilon_remain[train_dataset_name][identifier] > 0.0:
                self.sub_train_datasetidentifier_2_dataset_status[train_dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED

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
        if operator == "dataset":
            if origin_status == JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL:
                # print("in: dataset 1")
                update_path = JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_ALLSCHED
                new_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif operator == "significance":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_SIGNIFICANCE
                new_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
        elif operator == "failed":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
            elif origin_status == JOB_STATUS_KEY.DONE_ALL_SCHED:
                update_path = JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
            elif origin_status == JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL:
                update_path =JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
        elif operator == "recoming":
            if origin_status == JOB_STATUS_KEY.DONE_ALL_SCHED:
                update_path = JOB_STATUS_UPDATE_PATH.ALLSCHED_2_NOSCHED
                new_status = JOB_STATUS_KEY.NO_SCHE
            elif origin_status == JOB_STATUS_KEY.RUNNING:
                update_path = JOB_STATUS_UPDATE_PATH.RUNNING_2_NOSCHED
                new_status = JOB_STATUS_KEY.NO_SCHE
        return update_path, new_status

    def get_job_status_update_origin_target(self, status_update_path):
        if status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_GPUSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_GPU_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_DATASETSCHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_DATASET_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_SIGNIFICANCE:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.FAILED

        elif status_update_path == JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
            target_status = JOB_STATUS_KEY.FAILED

        elif status_update_path == JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.ALLSCHED_2_NOSCHED:
            origin_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            target_status = JOB_STATUS_KEY.NO_SCHE

        elif status_update_path == JOB_STATUS_UPDATE_PATH.RUNNING_2_NOSCHED:
            origin_status = JOB_STATUS_KEY.RUNNING
            target_status = JOB_STATUS_KEY.NO_SCHE

        return origin_status, target_status

    def get_job_datablock_significance_sync(self, all_significance_state):
        device_index = 0
        sub_train_index_2_significance = {
            index: None for index, _ in enumerate(all_significance_state)
        }
        for index, state in enumerate(all_significance_state):
            result_d = self.significance_policy.get_job_datablock_significance_sync(state)
            sub_train_index_2_significance[index] = result_d
        # 将所有为None的提取出来, 异步请求一下
        async_indexes = [k for k, v in sub_train_index_2_significance.items() if v is None]
        for index in async_indexes:
            print("[WARNING] enter into get_job_datablock_significance_async!")
            result_d = self.significance_policy.get_job_datablock_significance_async(all_significance_state[index], device_index)
            sub_train_index_2_significance[index] = result_d
        sub_train_datasetidentifier_2_significance = {
            all_significance_state[key]["sub_train_key_ids"][0]: value for key, value in sub_train_index_2_significance.items()
        }
        return sub_train_datasetidentifier_2_significance

    def calculate_significance_for_nosched_jobs(self):
        all_no_sche_jobs_copy = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE])
        if len(all_no_sche_jobs_copy) <= 0:
            return 
        for job_id in all_no_sche_jobs_copy:
            if job_id in self.jobid_2_significance:
                continue
            test_dataset_name = self.jobid_2_test_dataset_name[job_id]
            sub_test_key_ids = self.jobid_2_sub_test_key_ids[job_id]

            job_target_train_dataset_name = self.jobid_2_train_dataset_name[job_id]

            if job_target_train_dataset_name in self.sub_train_datasetidentifier_2_dataset_status:
                all_significance_state = []
                for sub_train_key_id in self.sub_train_datasetidentifier_2_dataset_status[job_target_train_dataset_name]:
                    significance_state = {
                        "test_dataset_name": test_dataset_name,
                        "sub_test_key_ids": sub_test_key_ids,
                        "train_dataset_name": job_target_train_dataset_name,
                        "sub_train_key_ids": [sub_train_key_id]
                    }
                    all_significance_state.append(significance_state)
            self.jobid_2_significance[job_id] = self.get_job_datablock_significance_sync(all_significance_state)
            self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.NO_SCHE, JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL)

    def sched_dataset_for_done_significance_cal_jobs(self):
        all_done_sig_cal_jobs_copy = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL])
        if len(all_done_sig_cal_jobs_copy) <= 0:
            return
        job_id_2_dataset_name = {job_id: self.jobid_2_train_dataset_name[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_target_epsilon_require = {job_id: self.jobid_2_target_epsilon[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_target_datablock_selected_num = {job_id: self.jobid_2_datablock_selected_num[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_job_priority_weight = {job_id: self.jobid_2_priority_weight[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_test_dataset_name = {job_id: self.jobid_2_test_dataset_name[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_sub_test_key_ids = {job_id: self.jobid_2_sub_test_key_ids[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_significance = {job_id: self.jobid_2_significance[job_id] for job_id in all_done_sig_cal_jobs_copy}
        job_id_2_arrival_index = {job_id: self.jobid_2_arrival_index[job_id] for job_id in all_done_sig_cal_jobs_copy}
        
        # 为没有决定分配方案的任务决定分配方案
        job_2_selected_datablock_identifiers, calcu_compare_epsilon = \
            self.get_scheduling_datablock_result(self.assignment_policy, job_id_2_dataset_name, 
                job_id_2_target_epsilon_require, job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                job_id_2_test_dataset_name, job_id_2_sub_test_key_ids, job_id_2_significance, job_id_2_arrival_index)
        success_sched_job_ids = set()
        if len(job_2_selected_datablock_identifiers) > 0:
            print("Jobs selected datablock identifiers: {}".format(job_2_selected_datablock_identifiers))
            for temp_job_id, identifier in job_2_selected_datablock_identifiers:
                if temp_job_id not in self.jobid_2_sub_train_key_ids:
                    self.jobid_2_sub_train_key_ids[temp_job_id] = []
                consume_epsilon = self.jobid_2_origininfo[temp_job_id]["EPSILON"]
                dataset_name = job_id_2_dataset_name[temp_job_id]
                if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] >= consume_epsilon:
                    self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] -= consume_epsilon # calcu_compare_epsilon
                    self.jobid_2_sub_train_key_ids[temp_job_id].append(identifier)
                    success_sched_job_ids.add(temp_job_id)
                if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] <= 0.0:
                    self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.EXHAUST
                    self.sub_train_datasetidentifier_2_exhausted_time[dataset_name][identifier] = time.time()
        
        need_failed_job = copy.deepcopy(all_done_sig_cal_jobs_copy)
        for temp_job_id in success_sched_job_ids:
            status_update_path, target_status = self.get_target_job_status_update_path_and_status(temp_job_id, "dataset")
            origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(temp_job_id, origin_status, target_status)
            need_failed_job.remove(temp_job_id)

            self.job_add_to_history(temp_job_id)

        for temp_job_id in need_failed_job:
            print("failed job scheduling [{}]".format(temp_job_id))
            # TODO(xlc): 这里不应该直接设置为Failed状态, 而是考虑max_time的情况, 决定是否将任务放到NO_SCHED的状态, 同时需要知道模型的最新保存位置?
            status_update_path, target_status = self.get_target_job_status_update_path_and_status(temp_job_id, "failed")
            origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(temp_job_id, origin_status, target_status)

            self.job_add_to_history(temp_job_id)

    def placement_dispatch(self, summary_writer_path):
        # 放置任务
        all_done_all_sched_jobs_copy = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED])
        if len(all_done_all_sched_jobs_copy) <= 0:
            return 
        args = []
        for job_id in all_done_all_sched_jobs_copy:
            origin_info = self.jobid_2_origininfo[job_id]
            worker_dataset_config = {
                "train_dataset_name": self.jobid_2_train_dataset_name[job_id],
                "test_dataset_name": self.jobid_2_test_dataset_name[job_id],
                "sub_train_key_ids": self.jobid_2_sub_train_key_ids[job_id],
                "sub_test_key_ids": self.jobid_2_sub_test_key_ids[job_id]
            }
            # 直接根据当前的空挡状态获取worker_identifier
            # worker_identifier = self.jobid_2_gputarget[job_id]
            gpuidentifier_enable_status = [k for k, v in self.gpuidentifier_2_jobinstance_oneshot.items() if v == None]
            if len(gpuidentifier_enable_status) > 0:
                gpu_identifer = random.choice(gpuidentifier_enable_status)
                if self.gpuidentifier_2_gpu_status[gpu_identifer] == True:
                    self.gpuidentifier_2_jobinstance_oneshot[gpu_identifer] = job_id
                    self.jobid_2_gputarget[job_id] = gpu_identifer
                    worker_ip, worker_gpu_id = self.get_worker_identifier_detail(gpu_identifer)
                    worker_port = self.workerip_2_ports[worker_ip]
                    logging_file_path = ""
                    args.append([job_id, origin_info, self.update_sched_epoch_num, worker_ip, worker_port, worker_gpu_id, worker_dataset_config, summary_writer_path, logging_file_path])
                    self.jobid_2_started_time[job_id] = time.time()
                    self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.DONE_ALL_SCHED, JOB_STATUS_KEY.RUNNING)
        if len(args) > 0:
            # self.report_status("after placement_dispatch all jobs")
            # 转置
            final_args = [[row[i] for row in args] for i in range(len(args[0]))]
            print("after placement_dispatch all job ids: [{}]".format(final_args[0]))
            with ThreadPoolExecutor(max_workers=len(args)) as pool:
                pool.map(DL_server_do_jobs, *final_args)

    def sched_dispatch_start(self, scheduler_update_sleep_time):
        def thread_func_timely_schedule(scheduler_update_sleep_time):
            time.sleep(scheduler_update_sleep_time)
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.sched_dataset_for_done_significance_cal_jobs()
                time.sleep(scheduler_update_sleep_time)
            print("Thread [thread_func_timely_schedule] finished!")
        p = threading.Thread(target=thread_func_timely_schedule, args=(scheduler_update_sleep_time, ), daemon=True)
        self.sched_thread = p
        p.start()
        print("Thread [thread_func_timely_schedule] started!")

    def cal_significance_dispatch_start(self, cal_significance_sleep_time):
        def thread_func_timely_cal_significance(cal_significance_sleep_time):
            time.sleep(cal_significance_sleep_time)
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.calculate_significance_for_nosched_jobs()
                time.sleep(cal_significance_sleep_time)
            print("Thread [thread_func_timely_cal_significance] finished!")
        p = threading.Thread(target=thread_func_timely_cal_significance, args=(cal_significance_sleep_time, ), daemon=True)
        self.cal_significance_thread = p
        p.start()
        print("Thread [thread_func_timely_cal_significance] started!")

    def placement_dispatch_start(self, placement_sleep_time, summary_writer_path):
        def thread_func_timely_placement(placement_sleep_time, summary_writer_path):
            time.sleep(placement_sleep_time)
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.placement_dispatch(summary_writer_path)
                time.sleep(placement_sleep_time)
            print("Thread [thread_func_timely_placement] finished!")
        p = threading.Thread(target=thread_func_timely_placement, args=(placement_sleep_time, summary_writer_path), daemon=True)
        self.placement_thread = p
        p.start()
        print("Thread [thread_func_timely_placement] started!")

    def schd_end(self):
        self.all_finished = True
        self.sched_thread = None
        self.cal_significance_thread = None
        self.placement_thread = None
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

    def sched_update_assignment_policy(self, assignment_policy, assignment_args):
        if assignment_policy == "PBGPolicy":
            comparison_cost_epsilon, comparison_z_threshold, L, U = assignment_args
            policy_item = PBGPolicy(comparison_cost_epsilon, comparison_z_threshold, L, U, self.logger)
        elif assignment_policy == "HISPolicy":
            beta = assignment_args
            policy_item = HISPolicy(beta, self.logger)
        elif assignment_policy == "DPFHISPolicy":
            beta, waiting_queue_capacity = assignment_args
            policy_item = DPFHISPolicy(beta, waiting_queue_capacity, self.logger)
        elif assignment_policy == "SagePolicy":
            policy_item = SagePolicy(self.logger)
        elif assignment_policy == "OfflinePolicy":
            policy_item = OfflinePolicy(self.logger)
        self.assignment_policy = policy_item

    def sched_update_significance_policy(self, significance_policy):
        if significance_policy == "HISOTDDPolicy":
            self.significance_policy = HISOTDDPolicy()
        elif significance_policy == "TempPolicy":
            self.significance_policy = TempPolicy()
        
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