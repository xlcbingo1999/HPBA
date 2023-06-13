import zerorpc
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing

import copy
import numpy as np
import random
import math
import argparse
import os
from utils.data_loader import fetch_new_dataset
from utils.get_profiler_significance import get_profiler_selection_result
from utils.global_functions import FAILED_RESULT_KEY, JOB_STATUS_KEY, JOB_STATUS_UPDATE_PATH, DATASET_STATUS_KEY, EVENT_KEY, get_zerorpc_client
from utils.global_variable import GPU_PATH, RESULT_PATH, DATASET_PATH
from utils.logging_tools import get_logger

from policies.PBG import PBGPolicy
from policies.PBGMix import PBGMixPolicy
from policies.SagewithRemain import SagewithRemainPolicy
from policies.BestFitwithRemain import BestFitwithRemainPolicy
from policies.HISwithOrderProVersion import HISwithOrderProVersionPolicy
from policies.IterativeHISwithOrderProVersion import IterativeHISwithOrderProVersionPolicy
from policies.Offline import OfflinePolicy

from significance_policies.Temp import TempPolicy
from significance_policies.OTDD import OTDDPolicy


from functools import reduce

import json
import time
import fcntl
import sys
from queue import PriorityQueue


def DL_server_do_jobs(job_id, origin_info, sched_epsilon_one_siton_run, siton_run_epoch_num, begin_epoch_num, worker_ip, worker_port, worker_gpu_id, 
                    worker_dataset_config, model_save_path, summary_writer_path, summary_writer_key, logging_file_path, final_significance, simulation_flag):
    client = zerorpc.Client()
    client.connect("tcp://{}:{}".format(worker_ip, worker_port))
    
    client.begin_job(job_id, worker_gpu_id, worker_dataset_config, origin_info, 
                    sched_epsilon_one_siton_run, begin_epoch_num, siton_run_epoch_num, 
                    model_save_path, summary_writer_path, summary_writer_key, logging_file_path, 
                    final_significance, simulation_flag)

class SchedEvent(object):
    def __init__(self, priority, event_key, metadata):
        self.priority = priority
        self.event_key = event_key
        self.metadata = metadata
    
    def __lt__(self, other): 
        return self.priority < other.priority
                   
    def __str__(self):
        return '(' + str(self.priority)+',\'' + str(self.event_key) + '\')'

class Scheduler_server(object):
    def __init__(self, sched_ip, sched_port):
        self.sched_ip = sched_ip
        self.sched_port = sched_port

        self.simulation = False
        self.simulation_index = 0
        self.simulation_queue = PriorityQueue()
        self.simulation_global_time = 0.0
        self.testbed_start_time = 0.0
        self.testbed_current_time = 0.0

        self.all_or_nothing_flag = False
        self.enable_waiting_flag = False

        self.all_stop = False
        self.all_finished = True

        self.all_testbed_thread = None
        self.sched_thread = None
        self.cal_significance_thread = None
        self.placement_thread = None
        self.gpu_thread = None

        self.finished_job_to_dispatcher_list = []
        self.finished_job_to_dispatcher_thread = None
        self.failed_job_to_dispatcher_list = []
        self.failed_job_to_dispatcher_thread = None
        
        self.workerip_2_ports = {}
        self.gpuidentifier_2_jobinstances = {}
        
        self.sub_train_datasetidentifier_2_dataset_status = {} # 这里必须是一个可以伸缩的map
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_arrival_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.sub_train_datasetidentifier_2_train_type = {}
        self.test_datasetname_2_metadata = {}
        
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {
            JOB_STATUS_KEY.SIMULATION_NO_SUMBIT: [], 
            JOB_STATUS_KEY.NO_SCHE: [], 
            JOB_STATUS_KEY.WAITING: [],
            JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL: [],
            JOB_STATUS_KEY.DONE_ALL_SCHED: [],
            JOB_STATUS_KEY.RUNNING: [], 
            JOB_STATUS_KEY.FINISHED: [],
            JOB_STATUS_KEY.FAILED: []
        }
        
        self.finished_update_init_history_jobs = False
        self.jobid_2_results = {}
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}

        # self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

        self.jobid_2_target_siton_run_epsilon = {}
        self.jobid_2_sched_siton_run_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_priority_weight = {}

        self.jobid_2_arrival_time = {}
        self.jobid_2_started_time = {}
        self.jobid_2_finished_time = {}
        self.jobid_2_train_dataset_name = {}
        self.jobid_2_sub_train_key_ids = {}
        self.jobid_2_datablock_selected_num = {}
        self.jobid_2_test_dataset_name = {}
        self.jobid_2_sub_test_key_id = {}
        self.jobid_2_target_significance = {}
        self.jobid_2_real_significance = {}
        self.jobid_2_arrival_index = {}
        self.jobid_2_typeid = {}
        self.min_significance_epsilon_ratio = float('inf')
        self.max_significance_epsilon_ratio = -float('inf')
        # self.jobid_2_recoming_min_time = {}
        # self.recoming_time_interval = 5

        self.jobid_2_siton_run_epoch_num = {}
        self.jobid_2_target_siton_run_num = {}
        self.jobid_2_success_siton_run_num = {}
        self.jobid_2_failed_siton_run_num = {}
        self.jobid_2_max_operate_siton_run_num = {}
        self.jobid_2_current_operate_siton_run_index = {}

        self.jobid_2_need_instantly_recoming = {}

        self.jobid_2_model_save_path = {}
        self.jobid_2_logging_file_path = {}
        self.jobid_2_summary_writer_key = {}

        self.pipeline_sequence_all_num = 0
        self.job_request_all_num = 0
        self.global_job_arrival_index = 0
        self.config_max_operate_siton_run_num = 0
        self.max_gpu_fuzai = 0

        self.dataset_name = ""
        self.dataset_config_name = ""

        self.assignment_policy = None
        self.significance_policy = None

        self.seed = 1234
        self.model_save_path = ""
        self.all_logger_path = ""
        self.summary_writer_path = ""
        self.sched_logger = None

    def initialize_sched_configs(self, simulation, simulation_index, 
                                seed, current_test_all_dir, 
                                all_or_nothing_flag, enable_waiting_flag, 
                                pipeline_sequence_all_num, job_request_all_num, 
                                config_max_operate_siton_run_num,
                                dataset_name, dataset_config_name, max_gpu_fuzai):
        # simulation
        self.simulation = simulation
        self.simulation_index = simulation_index

        # seed
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed+1)

        # logging path
        self.model_save_path = '{}/{}'.format(RESULT_PATH, current_test_all_dir)
        self.all_logger_path = '{}/{}'.format(RESULT_PATH, current_test_all_dir)
        self.summary_writer_path = '{}/{}'.format(RESULT_PATH, current_test_all_dir)

        sched_logger_path = '{}/DL_sched_{}.log'.format(self.all_logger_path, simulation_index)
        self.sched_logger = get_logger(sched_logger_path, sched_logger_path, enable_multiprocess=True)

        # all_or_nothing / waiting
        self.all_or_nothing_flag = all_or_nothing_flag
        self.enable_waiting_flag = enable_waiting_flag
        self.pipeline_sequence_all_num = pipeline_sequence_all_num
        self.job_request_all_num = job_request_all_num
        self.config_max_operate_siton_run_num = config_max_operate_siton_run_num
        self.max_gpu_fuzai = max_gpu_fuzai

        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name

    def check_all_finished_or_failed(self):
        return (len(self.status_2_jobid[JOB_STATUS_KEY.SIMULATION_NO_SUMBIT]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]) <= 0 
            and len(self.status_2_jobid[JOB_STATUS_KEY.WAITING]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED]) <= 0
            and len(self.status_2_jobid[JOB_STATUS_KEY.RUNNING]) <= 0
        )

    def get_all_no_finished_or_failed_jobs(self):
        return self.status_2_jobid[JOB_STATUS_KEY.SIMULATION_NO_SUMBIT] + \
            self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE] + \
            self.status_2_jobid[JOB_STATUS_KEY.WAITING] + \
            self.status_2_jobid[JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL] + \
            self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED] + \
            self.status_2_jobid[JOB_STATUS_KEY.RUNNING]

    def clear_all(self):
        for worker_ip, worker_port in self.workerip_2_ports.items():
            client = get_zerorpc_client(worker_ip, worker_port)
            self.sched_logger.debug("xlc clear all worker_ip: {} worker_port: {}".format(worker_ip, worker_port))
            client.clear_all_jobs()

        self.simulation = False
        self.simulation_index = 0
        self.simulation_queue = PriorityQueue()
        self.simulation_global_time = 0.0
        self.testbed_start_time = 0.0
        self.testbed_current_time = 0.0

        self.all_or_nothing_flag = False
        self.enable_waiting_flag = False

        self.all_stop = False
        self.all_finished = True

        self.all_testbed_thread = None
        self.sched_thread = None
        self.cal_significance_thread = None
        self.placement_thread = None
        self.gpu_thread = None

        self.finished_job_to_dispatcher_list = []
        self.finished_job_to_dispatcher_thread = None
        self.failed_job_to_dispatcher_list = []
        self.failed_job_to_dispatcher_thread = None

        self.gpuidentifier_2_jobinstances = {}
        
        self.sub_train_datasetidentifier_2_dataset_status = {} # 这里必须是一个可以伸缩的map
        self.sub_train_datasetidentifier_2_dataset_metadata = {}
        self.sub_train_datasetidentifier_2_epsilon_capacity = {}
        self.sub_train_datasetidentifier_2_epsilon_remain = {}
        self.sub_train_datasetidentifier_2_arrival_time = {}
        self.sub_train_datasetidentifier_2_exhausted_time = {}
        self.sub_train_datasetidentifier_2_train_type = {}
        self.test_datasetname_2_metadata = {}
        
        self.jobid_2_status = {} # 0: no sche; 1: sched target decide; 2: runnning; 3: success finished; 4: failed;
        self.status_2_jobid = {
            JOB_STATUS_KEY.SIMULATION_NO_SUMBIT: [], 
            JOB_STATUS_KEY.NO_SCHE: [], 
            JOB_STATUS_KEY.WAITING: [],
            JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL: [],
            JOB_STATUS_KEY.DONE_ALL_SCHED: [],
            JOB_STATUS_KEY.RUNNING: [], 
            JOB_STATUS_KEY.FINISHED: [],
            JOB_STATUS_KEY.FAILED: []
        }
        
        self.finished_update_init_history_jobs = False
        self.jobid_2_results = {} # TODO-OK(xlc): multi
        self.jobid_2_origininfo = {}
        self.jobid_2_gputarget = {}

        # self.jobid_2_datasettargetconfig = {}
        self.jobid_2_trainconfig = {}

        self.jobid_2_target_siton_run_epsilon = {}
        self.jobid_2_sched_siton_run_epsilon = {}
        self.jobid_2_real_epsilon = {}
        self.jobid_2_priority_weight = {}

        self.jobid_2_arrival_time = {} # TODO-OK(xlc): multi
        self.jobid_2_started_time = {} # TODO-OK(xlc): multi
        self.jobid_2_finished_time = {} # TODO-OK(xlc): multi
        
        self.jobid_2_train_dataset_name = {}
        self.jobid_2_sub_train_key_ids = {} # TODO-OK(xlc): multi
        self.jobid_2_datablock_selected_num = {}
        self.jobid_2_test_dataset_name = {}
        self.jobid_2_sub_test_key_id = {}

        self.jobid_2_target_significance = {}
        self.jobid_2_real_significance = {} # TODO-OK(xlc): multi
        self.jobid_2_arrival_index = {} # TODO-OK(xlc): multi 每次作为一个子任务都要提交一个新的arrival_index
        self.jobid_2_typeid = {}
        # self.jobid_2_recoming_min_time = {}
        # self.recoming_time_interval = 5

        self.jobid_2_siton_run_epoch_num = {}
        self.jobid_2_target_siton_run_num = {}
        self.jobid_2_success_siton_run_num = {}
        self.jobid_2_failed_siton_run_num = {}
        self.jobid_2_max_operate_siton_run_num = {}
        self.jobid_2_current_operate_siton_run_index = {}
        
        self.jobid_2_need_instantly_recoming = {}

        self.jobid_2_model_save_path = {}
        self.jobid_2_logging_file_path = {}
        self.jobid_2_summary_writer_key = {}

        self.pipeline_sequence_all_num = 0
        self.job_request_all_num = 0 # TODO(xlc): 这个值应该直接变成了任务的总数量, 注意在policy里面需要简单修改一下!
        self.global_job_arrival_index = 0
        self.config_max_operate_siton_run_num = 0
        self.max_gpu_fuzai = 0

        self.dataset_name = ""
        self.dataset_config_name = ""

        self.min_significance_epsilon_ratio = float('inf')
        self.max_significance_epsilon_ratio = -float('inf')

        self.assignment_policy = None
        self.significance_policy = None

        self.seed = 1234
        self.model_save_path = ""
        self.all_logger_path = ""
        self.summary_writer_path = ""
        self.sched_logger = None 

    def stop_all(self):
        print("sched stop_all")
        for worker_ip, worker_port in self.workerip_2_ports.items():
            client = get_zerorpc_client(worker_ip, worker_port)
            client.stop_all()
        self.all_stop = True

    def get_zerorpc_client(self, ip, port, timeout=500):
        tcp_ip_port = "tcp://{}:{}".format(ip, port)
        client = zerorpc.Client(timeout=timeout)
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
            self.sched_logger.info("success add datset[{}] with {} blocks".format(init_dataset_name, len(dataset_identifier_2_capacity_map)))
            if init_dataset_name not in self.sub_train_datasetidentifier_2_dataset_status:
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_arrival_time[init_dataset_name] = {}
                self.sub_train_datasetidentifier_2_exhausted_time[init_dataset_name] = {}
            
            for identifier in dataset_identifier_2_capacity_map:
                if identifier not in init_subtrain_datasets_map[init_dataset_name]:
                    self.sched_logger.warning("[warning] {} not in dataset config!".format(identifier))
                    continue
                if identifier in self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name]:
                    self.sched_logger.warning("[warning] {} already in dataset config!".format(identifier))
                    continue
                self.sub_train_datasetidentifier_2_dataset_status[init_dataset_name][identifier] = DATASET_STATUS_KEY.SUBMITED
                self.sub_train_datasetidentifier_2_dataset_metadata[init_dataset_name][identifier] = init_subtrain_datasets_map[init_dataset_name][identifier]
                self.sub_train_datasetidentifier_2_epsilon_capacity[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                self.sub_train_datasetidentifier_2_epsilon_remain[init_dataset_name][identifier] = dataset_identifier_2_capacity_map[identifier]
                if self.simulation:
                    self.sub_train_datasetidentifier_2_arrival_time[init_dataset_name][identifier] = self.simulation_global_time
                else:
                    self.sub_train_datasetidentifier_2_arrival_time[init_dataset_name][identifier] = self.testbed_current_time
                if self.assignment_policy.need_history:
                    # 这里必须对所有的offline和online历史记录都去计算一下新的significance!
                    offline_history_job_informations = self.assignment_policy.pull_offline_history_from_assignment_policy(
                        ["offline_history_job_test_dataset_name", "offline_history_job_sub_test_key_id",
                         "offline_history_job_train_dataset_name", "offline_history_job_type_id",
                         "offline_history_job_significance", "offline_history_job_model_name"]
                    )
                    for offline_index in range(len(offline_history_job_informations["offline_history_job_significance"])):
                        test_dataset_name = offline_history_job_informations["offline_history_job_test_dataset_name"][offline_index]
                        sub_test_key_id = offline_history_job_informations["offline_history_job_sub_test_key_id"][offline_index]
                        train_dataset_name = offline_history_job_informations["offline_history_job_train_dataset_name"][offline_index]
                        type_id = offline_history_job_informations["offline_history_job_type_id"][offline_index]
                        model_name = offline_history_job_informations["offline_history_job_model_name"][offline_index]
                        significance_state = [{
                            "test_dataset_name": test_dataset_name,
                            "sub_test_key_id": sub_test_key_id,
                            "train_dataset_name": train_dataset_name,
                            "sub_train_key_id": identifier,
                            "model_name": model_name
                        }]
                        result_d_map = self.get_job_datablock_significance_sync(type_id, significance_state)
                        offline_history_job_informations["offline_history_job_significance"][offline_index].update(result_d_map)
                    self.assignment_policy.update_offline_history_job_significance_to_assignment_policy(offline_history_job_informations["offline_history_job_significance"])

                    online_history_job_informations = self.assignment_policy.pull_online_history_from_assignment_policy(
                        ["online_history_job_test_dataset_name", "online_history_job_sub_test_key_id",
                         "online_history_job_train_dataset_name", "online_history_job_type_id",
                         "online_history_job_significance", "online_history_job_model_name"]
                    )
                    for online_index in range(len(online_history_job_informations["online_history_job_significance"])):
                        test_dataset_name = online_history_job_informations["online_history_job_test_dataset_name"][online_index]
                        sub_test_key_id = online_history_job_informations["online_history_job_sub_test_key_id"][online_index]
                        train_dataset_name = online_history_job_informations["online_history_job_train_dataset_name"][online_index]
                        type_id = online_history_job_informations["online_history_job_type_id"][online_index]
                        model_name = online_history_job_informations["online_history_job_model_name"][online_index]
                        significance_state = [{
                            "test_dataset_name": test_dataset_name,
                            "sub_test_key_id": sub_test_key_id,
                            "train_dataset_name": train_dataset_name,
                            "sub_train_key_id": identifier,
                            "model_name": model_name
                        }]
                        result_d_map = self.get_job_datablock_significance_sync(type_id, significance_state)
                        online_history_job_informations["online_history_job_significance"][online_index].update(result_d_map)
                    self.assignment_policy.update_online_history_job_significance_to_assignment_policy(online_history_job_informations["online_history_job_significance"])
                if self.enable_waiting_flag:
                    # self.sched_logger.debug(f"self.status_2_jobid[JOB_STATUS_KEY.WAITING]: {self.status_2_jobid[JOB_STATUS_KEY.WAITING]}")
                    for waiting_job_id in self.status_2_jobid[JOB_STATUS_KEY.WAITING]:
                        test_dataset_name = self.jobid_2_test_dataset_name[waiting_job_id]
                        sub_test_key_id = self.jobid_2_sub_test_key_id[waiting_job_id]
                        train_dataset_name = self.jobid_2_train_dataset_name[waiting_job_id]
                        model_name = self.jobid_2_origininfo[waiting_job_id]["model_name"]
                        type_id = self.jobid_2_typeid[waiting_job_id]
                        significance_state = [{
                            "test_dataset_name": test_dataset_name,
                            "sub_test_key_id": sub_test_key_id,
                            "train_dataset_name": train_dataset_name,
                            "sub_train_key_id": identifier,
                            "model_name": model_name
                        }]
                        result_d_map = self.get_job_datablock_significance_sync(type_id, significance_state)
                        
                        self.jobid_2_target_significance[waiting_job_id].update(result_d_map)
                        
        self.sched_logger.info(f"success add new datablocks with num[{len(init_subtrain_datasets_map)}]: {init_subtrain_datasets_map}")
    
    # def init_jobs_all_sequence_num(self, init_pipelines_all_sequence_num):
    #     self.pipeline_sequence_all_num = init_pipelines_all_sequence_num
    #     self.sched_logger.info("dispatcher init job_all_seq_num: {}".format(self.pipeline_sequence_all_num))

    def simulation_preview_no_submit_job(self, job_id):
        self.jobid_2_status[job_id] = JOB_STATUS_KEY.SIMULATION_NO_SUMBIT
        self.status_2_jobid[JOB_STATUS_KEY.SIMULATION_NO_SUMBIT].append(job_id)

    def update_jobs(self, jobs_detail_map): # 每次可以增加一批任务
        for id in jobs_detail_map:
            origin_info = jobs_detail_map[id]
            if self.simulation:
                update_path, _ = self.get_target_job_status_update_path_and_status(id, "simulation_submit")
                origin_status, new_status = self.get_job_status_update_origin_target(update_path)
                self.sche_reflash_job_status(id, origin_status, new_status)
                # self.report_status("after update job in simulation")
            else:
                if id in self.jobid_2_status:
                    self.sched_logger.warning("Waring: job {} has existed!".format(id))
                    continue
                self.jobid_2_status[id] = JOB_STATUS_KEY.NO_SCHE
                self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE].append(id)
            
            self.jobid_2_results[id] = [] # SOLVE(xlc): 修改为[]
            self.jobid_2_origininfo[id] = origin_info
            self.jobid_2_gputarget[id] = None
            self.jobid_2_trainconfig[id] = {}
            
            target_epsilon_consume = origin_info["EPSILON"] * origin_info["SITON_RUN_EPOCH_NUM"] # 单次消耗的隐私预算量
            self.jobid_2_target_siton_run_epsilon[id] = target_epsilon_consume
            self.jobid_2_sched_siton_run_epsilon[id] = 0
            self.jobid_2_real_epsilon[id] = 0
            self.jobid_2_priority_weight[id] = origin_info["priority_weight"]

            self.jobid_2_arrival_time[id] = []
            if self.simulation:
                self.jobid_2_arrival_time[id].append(self.simulation_global_time)
            else:
                self.jobid_2_arrival_time[id].append(self.testbed_current_time)
            self.jobid_2_started_time[id] = []
            self.jobid_2_finished_time[id] = []

            train_dataset_name = origin_info["train_dataset_name"]
            self.jobid_2_train_dataset_name[id] = train_dataset_name
            self.jobid_2_sub_train_key_ids[id] = []
            self.jobid_2_datablock_selected_num[id] = origin_info["datablock_select_num"]
            test_dataset_name = origin_info["test_dataset_name"]
            sub_test_key_id = origin_info["sub_test_key_id"]
            self.jobid_2_test_dataset_name[id] = test_dataset_name
            self.jobid_2_sub_test_key_id[id] = sub_test_key_id
            
            self.jobid_2_target_significance[id] = {}
            self.jobid_2_real_significance[id] = []
            self.jobid_2_arrival_index[id] = []
            self.jobid_2_arrival_index[id].append(self.global_job_arrival_index) 
            self.global_job_arrival_index += 1
            self.jobid_2_typeid[id] = origin_info["job_type"]
            

            self.jobid_2_siton_run_epoch_num[id] = origin_info["SITON_RUN_EPOCH_NUM"]
            self.jobid_2_target_siton_run_num[id] = int(np.ceil(origin_info["TARGET_EPOCHS"] / origin_info["SITON_RUN_EPOCH_NUM"]))
            self.jobid_2_success_siton_run_num[id] = 0
            self.jobid_2_failed_siton_run_num[id] = 0
            self.jobid_2_max_operate_siton_run_num[id] = self.config_max_operate_siton_run_num
            self.jobid_2_current_operate_siton_run_index[id] = 0
            
            self.jobid_2_need_instantly_recoming[id] = False

            self.jobid_2_model_save_path[id] = self.model_save_path + "/{}.pt".format(id)
            self.jobid_2_logging_file_path[id] = self.all_logger_path + "/{}.log".format(id)
            self.jobid_2_summary_writer_key[id] = "{}".format(id)

        self.sched_logger.info(f"success add new jobs with num[{len(jobs_detail_map)}]: {jobs_detail_map}")

    def update_history_jobs(self, history_jobs_map):
        offline_history_job_priority_weights = []
        offline_history_job_budget_consumes = []
        offline_history_job_target_selected_num = []
        offline_history_job_train_dataset_name = []
        offline_history_job_test_dataset_name = []
        offline_history_job_sub_test_key_id = []
        offline_history_job_significance = []
        offline_history_job_type_id = []
        offline_history_job_arrival_time = []
        offline_history_job_model_name = []
        for id in sorted(history_jobs_map):
            # target_job_epoch_num = history_jobs_map[id]["TARGET_EPOCHS"]
            offline_history_job_priority_weights.append(history_jobs_map[id]["priority_weight"])

            target_epsilon_consume = history_jobs_map[id]["EPSILON"] * history_jobs_map[id]["TARGET_EPOCHS"]
            offline_history_job_budget_consumes.append(target_epsilon_consume)

            target_selected_num = history_jobs_map[id]["datablock_select_num"]
            offline_history_job_target_selected_num.append(target_selected_num)

            train_dataset_name = history_jobs_map[id]["train_dataset_name"]
            offline_history_job_train_dataset_name.append(train_dataset_name)

            test_dataset_name = history_jobs_map[id]["test_dataset_name"]
            offline_history_job_test_dataset_name.append(test_dataset_name)

            sub_test_key_id = history_jobs_map[id]["sub_test_key_id"]
            offline_history_job_sub_test_key_id.append(sub_test_key_id)
            
            type_id = history_jobs_map[id]["job_type"]
            offline_history_job_type_id.append(type_id)

            arrival_time = history_jobs_map[id]["time"]
            offline_history_job_arrival_time.append(arrival_time)

            model_name = history_jobs_map[id]["model_name"]
            offline_history_job_model_name.append(model_name)
            
            if train_dataset_name in self.sub_train_datasetidentifier_2_dataset_status: # 当历史记录到来前还没有历史记录的时候就需要处理
                all_significance_state = []
                for sub_train_key_id in self.sub_train_datasetidentifier_2_dataset_status[train_dataset_name]:
                    significance_state = {
                        "test_dataset_name": test_dataset_name,
                        "sub_test_key_id": sub_test_key_id,
                        "train_dataset_name": train_dataset_name,
                        "sub_train_key_id": sub_train_key_id,
                        "model_name": model_name
                    }
                    all_significance_state.append(significance_state)
                result_d_map = self.get_job_datablock_significance_sync(type_id, all_significance_state)
                offline_history_job_significance.append(result_d_map)
                
        self.sched_logger.info("success add all offline history jobs len(history_jobs_map): {}".format(len(history_jobs_map)))
        if self.assignment_policy.need_history:
            self.assignment_policy.push_offline_history_to_assignment_policy(
                offline_history_job_priority_weights,
                offline_history_job_budget_consumes,
                offline_history_job_target_selected_num,
                offline_history_job_train_dataset_name,
                offline_history_job_test_dataset_name,
                offline_history_job_sub_test_key_id,
                offline_history_job_type_id,
                offline_history_job_significance,
                offline_history_job_arrival_time,
                offline_history_job_model_name
            )
        self.finished_update_init_history_jobs = True

    def sched_simulation_start(self, subtrain_datasetidentifier_info, all_history_jobs_detail, all_jobs_detail):
        def thread_func_sched_simulation(subtrain_datasetidentifier_info, all_history_jobs_detail, all_jobs_detail):
            self.simulation_queue.put(SchedEvent(-0.1, EVENT_KEY.TEST_START, {}))
            for dataset_name in subtrain_datasetidentifier_info:
                for sub_train_dataset_identifier in subtrain_datasetidentifier_info[dataset_name]:
                    need_submit_time = subtrain_datasetidentifier_info[dataset_name][sub_train_dataset_identifier]["time"]
                    # self.sched_logger.info("[add dataset start dataset_name: {}; sub_train_dataset_identifier: {}]  need_submit_time: {}".format(
                    #     dataset_name, sub_train_dataset_identifier, need_submit_time
                    # ))
                    
                    event_info = {dataset_name: {sub_train_dataset_identifier: subtrain_datasetidentifier_info[dataset_name][sub_train_dataset_identifier]}}
                    self.simulation_queue.put(SchedEvent(need_submit_time, EVENT_KEY.DATABLOCK_ADD, event_info))
            
            self.simulation_queue.put(SchedEvent(-1, EVENT_KEY.HISTORY_JOB_SUBMIT, all_history_jobs_detail))
            
            for index in range(len(all_jobs_detail)):
                job_id, info = all_jobs_detail[index]
                
                need_submit_time = info["time"]
                # self.sched_logger.info("[add job start job_id: {}] need_submit_time: {}".format(job_id, need_submit_time))
                dispatch_jobs_detail = {}
                dispatch_jobs_detail[job_id] = info
                
                self.simulation_preview_no_submit_job(job_id)
                self.simulation_queue.put(SchedEvent(need_submit_time, EVENT_KEY.JOB_SUBMIT, dispatch_jobs_detail))
            self.sched_logger.info("thread_func_create_priority_queue finished!") 
            
            next_event = self.simulation_queue.get()
            next_time = next_event.priority 
            self.simulation_global_time = next_time
            waiting_for_end = False
            dispatchers = set()
            while True:
                if self.all_finished:
                    self.sched_logger.info("all_finished")
                    break
                if not waiting_for_end:
                    self.sched_logger.info("simulation_global_time[{}] next_event: {}".format(self.simulation_global_time, next_event))
                    if next_event.event_key == EVENT_KEY.JOB_SUBMIT:
                        self.update_jobs(next_event.metadata)
                        job_id, info = list(next_event.metadata.items())[0]
                        dispatchers.add((info["dispatcher_ip"], info["dispatcher_port"]))
                        self.simulation_queue.put(SchedEvent(self.simulation_global_time, EVENT_KEY.JOB_SIGCAL_SCHED_RUNNING, {}))
                    if next_event.event_key == EVENT_KEY.JOB_SIGCAL_SCHED_RUNNING:
                        self.calculate_significance_for_nosched_jobs()
                        self.sched_dataset_for_jobs(JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL)
                        if self.enable_waiting_flag:
                            self.sched_dataset_for_jobs(JOB_STATUS_KEY.WAITING)
                        self.placement_dispatch_for_allsched_jobs()
                    elif next_event.event_key == EVENT_KEY.HISTORY_JOB_SUBMIT:
                        self.update_history_jobs(next_event.metadata)
                    elif next_event.event_key == EVENT_KEY.DATABLOCK_ADD:
                        self.update_dataset(next_event.metadata)
                        need_instant_recoming_jobs = self.status_2_jobid[JOB_STATUS_KEY.WAITING]
                        for job_id in need_instant_recoming_jobs:
                            self.jobid_2_need_instantly_recoming[job_id] = True
                        self.simulation_queue.put(SchedEvent(self.simulation_global_time, EVENT_KEY.JOB_SIGCAL_SCHED_RUNNING, {}))
                    elif next_event.event_key == EVENT_KEY.TEST_START:
                        self.sched_logger.info("============= TEST_START =============")
                
                if self.check_all_finished_or_failed():
                    self.report_status("after check_all_finished_or_failed")
                    self.sched_logger.info("check_all_job state: finished_or_failed")
                    self.sched_end()
                    self.end_and_report_dispatchers_by_sched(dispatchers)
                else:
                    if not self.simulation_queue.empty():
                        next_event = self.simulation_queue.get()
                        next_time = next_event.priority 
                        self.simulation_global_time = next_time
                    else:
                        waiting_for_end = True
            self.sched_logger.info("thread_func_sched_simulation finished!") 
        self.sched_logger.info("thread_func_sched_simulation start!") 
        p = threading.Thread(target=thread_func_sched_simulation, args=(subtrain_datasetidentifier_info, all_history_jobs_detail, all_jobs_detail))
        p.start()

    def sche_reflash_job_status(self, job_id, origin_status, new_status):
        self.jobid_2_status[job_id] = new_status
        self.status_2_jobid[origin_status].remove(job_id)
        self.status_2_jobid[new_status].append(job_id)
        self.sched_logger.debug("id {} in origin [{}] -> new [{}]".format(job_id, origin_status, new_status))

    def job_add_to_history(self, job_id):
        job_siton_run_index = self.jobid_2_current_operate_siton_run_index[job_id]
        if self.assignment_policy.need_history:
            self.assignment_policy.push_online_history_to_assignment_policy(
                self.jobid_2_priority_weight[job_id], self.jobid_2_target_siton_run_epsilon[job_id],
                self.jobid_2_datablock_selected_num[job_id], self.jobid_2_train_dataset_name[job_id],
                self.jobid_2_test_dataset_name[job_id], self.jobid_2_sub_test_key_id[job_id], 
                self.jobid_2_typeid[job_id], self.jobid_2_target_significance[job_id],
                self.jobid_2_arrival_time[job_id][job_siton_run_index], self.jobid_2_origininfo[job_id]["model_name"]
            )
        self.sched_logger.debug("success add a new history job")

    def get_all_state_num(self):
        current_success_num = len(self.status_2_jobid[JOB_STATUS_KEY.FINISHED])
        current_failed_num = len(self.status_2_jobid[JOB_STATUS_KEY.FAILED])
        current_no_submit_num = len(self.status_2_jobid[JOB_STATUS_KEY.SIMULATION_NO_SUMBIT])
        current_no_sche_num = len(self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE])
        current_done_sig_num = len(self.status_2_jobid[JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL])
        current_done_sche_num = len(self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED])
        current_running_num = len(self.status_2_jobid[JOB_STATUS_KEY.RUNNING])
        current_recoming_num = len(self.status_2_jobid[JOB_STATUS_KEY.WAITING])
        return current_success_num, current_failed_num, current_no_submit_num, current_no_sche_num, \
            current_done_sig_num, current_done_sche_num, current_running_num, current_recoming_num

    def get_result_status(self):
        all_train_loss = 0.0
        all_train_accuracy = 0.0
        all_test_loss = 0.0
        all_test_accuracy = 0.0
        all_final_significance = 0.0
        for job_id in self.jobid_2_results:
            job_res = self.jobid_2_results[job_id]
            self.sched_logger.debug("job [{}] last result: {}".format(job_id, job_res))
            if len(job_res) <= 0:
                all_train_loss += 0.0
                all_train_accuracy += 0.0
                all_test_loss += 0.0
                all_test_accuracy += 0.0 # TODO(xlc): 这里应该加上init, 不过其实没训练的模型基本上应该也是不准的吧, 直接设置为0.0算了, 不象处理
                all_final_significance += 0.0
            else:
                last_job_res = job_res[-1]
                if "train_loss" in last_job_res:
                    all_train_loss += last_job_res["train_loss"]
                if "train_acc" in last_job_res:
                    all_train_accuracy += last_job_res["train_acc"]
                if "test_loss" in last_job_res:
                    all_test_loss += last_job_res["test_loss"]
                if "test_acc" in last_job_res:
                    all_test_accuracy += last_job_res["test_acc"]
                for sub_res in job_res:
                    if "final_significance" in sub_res:
                        all_final_significance += sub_res["final_significance"]
        return all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_final_significance 

    def report_status(self, location):
        self.sched_logger.debug("======== Scheduler Status in {} ========".format(location))
        current_success_num, current_failed_num, current_no_submit_num, current_no_sche_num, \
            current_done_sig_num, current_done_sche_num, current_running_num, current_recoming_num = \
            self.get_all_state_num()

        self.sched_logger.debug("current_success_num: {}; current_failed_num: {}; current_no_submit_num: {}; current_no_sche_num: {};".format(
            current_success_num, current_failed_num, current_no_submit_num, current_no_sche_num
        ))
        self.sched_logger.debug("current_done_sig_num: {}; current_done_sche_num: {}; current_running_num: {}; current_recoming_num: {};".format(
            current_done_sig_num, current_done_sche_num, current_running_num, current_recoming_num
        ))
        self.sched_logger.debug("======== result status ========")
        all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_final_significance = self.get_result_status()
        self.sched_logger.debug("all_test_jobs_num: {}".format(self.pipeline_sequence_all_num))
        self.sched_logger.debug("all_train_loss: {}".format(all_train_loss /  self.pipeline_sequence_all_num))
        self.sched_logger.debug("all_train_accuracy: {}".format(all_train_accuracy /  self.pipeline_sequence_all_num))
        self.sched_logger.debug("all_test_loss: {}".format(all_test_loss /  self.pipeline_sequence_all_num))
        self.sched_logger.debug("all_test_accuracy: {}".format(all_test_accuracy /  self.pipeline_sequence_all_num))
        self.sched_logger.debug("all_final_significance: {}".format(all_final_significance /  self.pipeline_sequence_all_num))
        
        if current_success_num > 0:
            self.sched_logger.debug("success_train_loss: {}".format(all_train_loss / current_success_num))
            self.sched_logger.debug("success_train_accuracy: {}".format(all_train_accuracy / current_success_num))
            self.sched_logger.debug("success_test_loss: {}".format(all_test_loss / current_success_num))
            self.sched_logger.debug("success_test_accuracy: {}".format(all_test_accuracy / current_success_num))
            self.sched_logger.debug("success_final_significance: {}".format(all_final_significance / current_success_num))

        
        self.sched_logger.debug("======== epsilon remain status =========")
        temp_iter_datasetidentifier_remain = self.sub_train_datasetidentifier_2_epsilon_remain.copy()
        for datasetname in temp_iter_datasetidentifier_remain:
            for datasetidentifier in temp_iter_datasetidentifier_remain[datasetname]:
                self.sched_logger.debug("sub_train_datasetidentifier_2_epsilon_remain[{}][{}]: {}".format(datasetname, datasetidentifier, temp_iter_datasetidentifier_remain[datasetname][datasetidentifier]))
        
        self.sched_logger.debug("======== PBG significance budget ratio =========")
        self.sched_logger.debug("min_significance_epsilon_ratio(L): {}".format(self.min_significance_epsilon_ratio))
        self.sched_logger.debug("max_significance_epsilon_ratio(U): {}".format(self.max_significance_epsilon_ratio))
        for job_id in self.jobid_2_target_siton_run_num:
            self.sched_logger.debug("job [{}] target_siton_run_num: {}".format(job_id, self.jobid_2_target_siton_run_num[job_id]))
            self.sched_logger.debug("job [{}] success_siton_run_num: {}".format(job_id, self.jobid_2_success_siton_run_num[job_id]))
            self.sched_logger.debug("job [{}] failed_siton_run_num: {}".format(job_id, self.jobid_2_failed_siton_run_num[job_id]))
            self.sched_logger.debug("job [{}] current_operate_siton_run_index: {}".format(job_id, self.jobid_2_current_operate_siton_run_index[job_id]))
        self.sched_logger.debug("==================================")

    def get_runtime_state(self, job_id_2_train_dataset_name, job_id_2_target_epsilon_require, 
                        job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                        job_id_2_test_dataset_name, job_id_2_sub_test_key_id, job_id_2_target_significance, 
                        job_id_2_arrival_index, job_id_2_arrival_time):
        state = {}
        state["job_id_2_train_dataset_name"] = job_id_2_train_dataset_name
        state["job_id_2_target_epsilon_require"] = job_id_2_target_epsilon_require
        state["job_id_2_target_datablock_selected_num"] = job_id_2_target_datablock_selected_num
        state["job_id_2_job_priority_weight"] = job_id_2_job_priority_weight
        state["job_id_2_test_dataset_name"] = job_id_2_test_dataset_name
        state["job_id_2_sub_test_key_id"] = job_id_2_sub_test_key_id
        state["job_id_2_significance"] = job_id_2_target_significance
        state["job_id_2_arrival_index"] = job_id_2_arrival_index
        state["job_id_2_arrival_time"] = job_id_2_arrival_time

        state["current_sub_train_datasetidentifier_2_epsilon_remain"] = copy.deepcopy(self.sub_train_datasetidentifier_2_epsilon_remain)
        state["current_sub_train_datasetidentifier_2_epsilon_capcity"] = copy.deepcopy(self.sub_train_datasetidentifier_2_epsilon_capacity)
        state["current_sub_train_datasetidentifier_2_arrival_time"] = copy.deepcopy(self.sub_train_datasetidentifier_2_arrival_time)
 
        # self.sched_logger.debug("get state current_sub_train_datasetidentifier_2_epsilon_remain: {}".format(self.sub_train_datasetidentifier_2_epsilon_remain))
        return state
    
    # def get_significance_state(self, policy, train_dataset_name, datablock_identifier, test_type, target_epsilon_consume):
    #     signficance_state = {}
    #     signficance_state["train_dataset_name"] = train_dataset_name
    #     signficance_state["datablock_identifier"] = datablock_identifier
    #     signficance_state["test_type"] = test_type
    #     if policy.name == "TempPolicy":
    #         signficance_state["epsilon_consume"] = target_epsilon_consume
    #     return signficance_state
    
    def get_scheduling_datablock_result_from_policy(self, job_id_2_dataset_name, job_id_2_target_epsilon_require, 
                                        job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                                        job_id_2_test_dataset_name, job_id_2_sub_test_key_id, 
                                        job_id_2_target_significance, job_id_2_arrival_index, 
                                        job_id_2_arrival_time):
        job_2_selected_datablock_identifiers = {}
        # 在这里接入算法?
        state = self.get_runtime_state(job_id_2_dataset_name, job_id_2_target_epsilon_require, 
                                    job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                                    job_id_2_test_dataset_name, job_id_2_sub_test_key_id, 
                                    job_id_2_target_significance, job_id_2_arrival_index, 
                                    job_id_2_arrival_time)
        job_2_selected_datablock_identifiers, sched_fail_job_ids, \
            selected_real_sched_epsilon_map, calcu_compare_epsilon, job_2_instant_recoming_flag = \
            self.assignment_policy.get_allocation(state, self.all_or_nothing_flag, self.enable_waiting_flag)
        # not_selected_datablock_identifiers = [tu[0] for tu in sub_train_sort[target_datablock_select_num:]]
        return job_2_selected_datablock_identifiers, sched_fail_job_ids, selected_real_sched_epsilon_map, calcu_compare_epsilon, job_2_instant_recoming_flag

    def push_success_scheduling_result_to_policy(self, success_datasetidentifier_2_consume_epsilon):
        self.assignment_policy.push_success_allocation(success_datasetidentifier_2_consume_epsilon)

    def finished_job_to_dispatcher(self, job_id, origin_info):
        self.sched_logger.info("job_id {} call finished_job_to_dispatcher".format(job_id))
        dispatcher_ip = origin_info["dispatcher_ip"]
        dispatcher_port = origin_info["dispatcher_port"]
        results = self.jobid_2_results[job_id]
        
        self.finished_job_to_dispatcher_list.append({
            "job_id": job_id,
            "dispatcher_ip": dispatcher_ip,
            "dispatcher_port": dispatcher_port,
            "results": results
        })

    def failed_job_to_dispatcher(self, job_id, origin_info):
        dispatcher_ip = origin_info["dispatcher_ip"]
        dispatcher_port = origin_info["dispatcher_port"]
        results = self.jobid_2_results[job_id]
        
        self.failed_job_to_dispatcher_list.append({
            "job_id": job_id,
            "dispatcher_ip": dispatcher_ip,
            "dispatcher_port": dispatcher_port,
            "results": results
        })

    def thread_finished_job_to_dispatcher_start(self):
        def thread_func_finished_job_to_dispatcher(sleep_time):
            while (not self.all_finished) and self.finished_update_init_history_jobs:
                while len(self.finished_job_to_dispatcher_list) > 0:
                    details = self.finished_job_to_dispatcher_list.pop(0)
                    job_id = details["job_id"]
                    dispatcher_ip = details["dispatcher_ip"]
                    dispatcher_port = details["dispatcher_port"]
                    results = details["results"]
                    
                    dispatcher_client = get_zerorpc_client(dispatcher_ip, dispatcher_port)
                    dispatcher_client.finished_job_callback(job_id, results)
                time.sleep(sleep_time)
            self.sched_logger.info("Thread [thread_func_finished_job_to_dispatcher] finished!")
        p = threading.Thread(target=thread_func_finished_job_to_dispatcher, args=(1,), daemon=True)
        self.finished_job_to_dispatcher_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_finished_job_to_dispatcher] started!")

    def thread_failed_job_to_dispatcher_start(self):
        def thread_func_failed_job_to_dispatcher(sleep_time):
            while (not self.all_finished) and self.finished_update_init_history_jobs:
                while len(self.failed_job_to_dispatcher_list) > 0:
                    details = self.failed_job_to_dispatcher_list.pop(0)
                    job_id = details["job_id"]
                    dispatcher_ip = details["dispatcher_ip"]
                    dispatcher_port = details["dispatcher_port"]
                    results = details["results"]

                    dispatcher_client = get_zerorpc_client(dispatcher_ip, dispatcher_port)
                    dispatcher_client.failed_job_callback(job_id, results)
                time.sleep(sleep_time)
            self.sched_logger.info("Thread [thread_func_failed_job_to_dispatcher] finished!")
        p = threading.Thread(target=thread_func_failed_job_to_dispatcher, args=(1,), daemon=True)
        self.failed_job_to_dispatcher_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_failed_job_to_dispatcher] started!")

    def worker_waiting_job_callback(self, job_id, origin_info):
        self.sched_logger.info(f"Waiting callback job_id: {job_id} => origin_info: {origin_info}")

        gpu_identifier = self.jobid_2_gputarget[job_id]
        self.jobid_2_gputarget[job_id] = None
        if gpu_identifier is not None and job_id in self.gpuidentifier_2_jobinstances[gpu_identifier]:
            self.gpuidentifier_2_jobinstances[gpu_identifier].remove(job_id)

        if self.enable_waiting_flag: # 只针对失败的场景
            sched_job_origin_state = self.jobid_2_status[job_id]
            if sched_job_origin_state != JOB_STATUS_KEY.WAITING:
                status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "wait")
                origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
                self.sche_reflash_job_status(job_id, origin_status, target_status)

            self.jobid_2_current_operate_siton_run_index[job_id] += 1
            self.jobid_2_failed_siton_run_num[job_id] += 1
            self.global_job_arrival_index += 1
            self.jobid_2_arrival_index[job_id].append(self.global_job_arrival_index)

            if self.jobid_2_current_operate_siton_run_index[job_id] >= self.jobid_2_max_operate_siton_run_num[job_id]:
                self.worker_failed_job_callback(job_id, origin_info, FAILED_RESULT_KEY.TRY_MAX_TIME)

    def worker_failed_job_callback(self, job_id, origin_info, failed_result_key):
        self.sched_logger.info("=========  Scheduler: Job Failed! ===========")
        self.sched_logger.info("job_id: {}".format(job_id))
        self.sched_logger.info("origin_info: {}".format(origin_info))
        self.sched_logger.info("failed_result_key: {}".format(failed_result_key))
        self.sched_logger.info("====================")

        gpu_identifier = self.jobid_2_gputarget[job_id]
        self.jobid_2_gputarget[job_id] = None
        if gpu_identifier is not None and job_id in self.gpuidentifier_2_jobinstances[gpu_identifier]:
            self.gpuidentifier_2_jobinstances[gpu_identifier].remove(job_id)
        
        status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "failed")
        origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
        self.sche_reflash_job_status(job_id, origin_status, target_status)
        if not self.simulation:
            self.failed_job_to_dispatcher(job_id, origin_info)
        
    def worker_runtime_failed_job_callback(self, job_id, origin_info, exception_log):
        self.sched_logger.info(f"worker_runtime_failed_job_callback job_id: {job_id} => exception_log: {exception_log}")
        gpu_identifier = self.jobid_2_gputarget[job_id]
        self.jobid_2_gputarget[job_id] = None
        if gpu_identifier is not None and job_id in self.gpuidentifier_2_jobinstances[gpu_identifier]:
            self.gpuidentifier_2_jobinstances[gpu_identifier].remove(job_id)
        
        if exception_log == "The privacy budget is too low.":
            self.worker_failed_job_callback(job_id, origin_info, FAILED_RESULT_KEY.RUNNING_FAILED)
        else:
            self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.RUNNING, JOB_STATUS_KEY.DONE_ALL_SCHED) # 恢复等待下一次调度即可

    def worker_finished_job_callback(self, job_id, origin_info, result):
        self.sched_logger.info(f"Temp Finished callback job_id: {job_id} => origin_info: {origin_info}; result: {result}")

        if self.simulation:
            self.jobid_2_finished_time[job_id].append(self.simulation_global_time)
        else:
            self.jobid_2_finished_time[job_id].append(self.testbed_current_time)

        # TODO(xlc): 获取job的current_epoch!
        self.jobid_2_results[job_id].append(result)
        self.jobid_2_real_epsilon[job_id] = result["epsilon_consume"]
        datablock_identifiers = self.jobid_2_sub_train_key_ids[job_id][-1]

        if self.significance_policy.need_update_backward:
            type_id = self.jobid_2_typeid[job_id]
            self.significance_policy.update_job_datablock_signficance_FAIR(type_id, datablock_identifiers, result)

        gpu_identifier = self.jobid_2_gputarget[job_id]
        self.jobid_2_gputarget[job_id] = None
        if gpu_identifier is not None and job_id in self.gpuidentifier_2_jobinstances[gpu_identifier]:
            self.gpuidentifier_2_jobinstances[gpu_identifier].remove(job_id)

        self.jobid_2_success_siton_run_num[job_id] += 1
        if self.enable_waiting_flag and result['test_acc'] < self.jobid_2_origininfo[job_id]["TAGRET_ACC"]:
            self.worker_waiting_job_callback(job_id, origin_info)
        else:
            self.sched_logger.info("=========  Scheduler: Job FINISHED! ===========")
            self.sched_logger.info("job_id: {}".format(job_id))
            self.sched_logger.info("origin_info: {}".format(origin_info))
            self.sched_logger.info("result: {}".format(result))
            self.sched_logger.info("====================")
            status_update_path, target_status = self.get_target_job_status_update_path_and_status(job_id, "finished")
            origin_status, target_status = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(job_id, origin_status, target_status)
            if not self.simulation:
                self.finished_job_to_dispatcher(job_id, origin_info)
                # self.report_status("finished job: {}".format(job_id))

    def get_target_job_status_update_path_and_status(self, job_id, operator):
        origin_status = self.jobid_2_status[job_id]
        update_path = None
        new_status = None
        if operator == "dataset":
            if origin_status == JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL:
                update_path = JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_ALLSCHED
                new_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            elif origin_status == JOB_STATUS_KEY.WAITING:
                update_path = JOB_STATUS_UPDATE_PATH.WAITING_2_ALLSCHED
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
                update_path = JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
            elif origin_status == JOB_STATUS_KEY.RUNNING:
                update_path = JOB_STATUS_UPDATE_PATH.RUNNING_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
            elif origin_status == JOB_STATUS_KEY.WAITING:
                update_path = JOB_STATUS_UPDATE_PATH.WAITING_2_FAILED
                new_status = JOB_STATUS_KEY.FAILED
        elif operator == "wait":
            if origin_status == JOB_STATUS_KEY.DONE_ALL_SCHED:
                update_path = JOB_STATUS_UPDATE_PATH.ALLSCHED_2_RECOMING
                new_status = JOB_STATUS_KEY.WAITING
            elif origin_status == JOB_STATUS_KEY.RUNNING:
                update_path = JOB_STATUS_UPDATE_PATH.RUNNING_2_RECOMING
                new_status = JOB_STATUS_KEY.WAITING
            elif origin_status == JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL:
                update_path = JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_RECOMING
                new_status = JOB_STATUS_KEY.WAITING
            elif origin_status == JOB_STATUS_KEY.NO_SCHE:
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_RECOMING
                new_status = JOB_STATUS_KEY.WAITING
        elif operator == "finished":
            if origin_status == JOB_STATUS_KEY.NO_SCHE:
                update_path = JOB_STATUS_UPDATE_PATH.NOSCHED_2_FINISHED
                new_status = JOB_STATUS_KEY.FINISHED
            elif origin_status == JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL:
                update_path = JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_FINISHED
                new_status = JOB_STATUS_KEY.FINISHED
            elif origin_status == JOB_STATUS_KEY.DONE_ALL_SCHED:
                update_path = JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FINISHED
                new_status = JOB_STATUS_KEY.FINISHED
            elif origin_status == JOB_STATUS_KEY.RUNNING:
                update_path = JOB_STATUS_UPDATE_PATH.RUNNING_2_FINISHED
                new_status = JOB_STATUS_KEY.FINISHED
        elif operator == "simulation_submit":
            if origin_status == JOB_STATUS_KEY.SIMULATION_NO_SUMBIT:
                update_path = JOB_STATUS_UPDATE_PATH.SIMULATION_NOSUBMIT_2_NOSHCED
                new_status = JOB_STATUS_KEY.NO_SCHE
        return update_path, new_status

    def get_job_status_update_origin_target(self, status_update_path):
        origin_status = None
        target_status = None
        # dataset
        if status_update_path == JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.WAITING_2_ALLSCHED:
            origin_status = JOB_STATUS_KEY.WAITING
            target_status = JOB_STATUS_KEY.DONE_ALL_SCHED
        # significance
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_SIGNIFICANCE:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
        # failed
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_FAILED:
            origin_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.RUNNING_2_FAILED:
            origin_status = JOB_STATUS_KEY.RUNNING
            target_status = JOB_STATUS_KEY.FAILED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.WAITING_2_FAILED:
            origin_status = JOB_STATUS_KEY.WAITING
            target_status = JOB_STATUS_KEY.FAILED
        # waiting
        elif status_update_path == JOB_STATUS_UPDATE_PATH.ALLSCHED_2_RECOMING:
            origin_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            target_status = JOB_STATUS_KEY.WAITING
        elif status_update_path == JOB_STATUS_UPDATE_PATH.RUNNING_2_RECOMING:
            origin_status = JOB_STATUS_KEY.RUNNING
            target_status = JOB_STATUS_KEY.WAITING
        elif status_update_path == JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_RECOMING:
            origin_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
            target_status = JOB_STATUS_KEY.WAITING
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_RECOMING:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.WAITING
        # finished
        elif status_update_path == JOB_STATUS_UPDATE_PATH.NOSCHED_2_FINISHED:
            origin_status = JOB_STATUS_KEY.NO_SCHE
            target_status = JOB_STATUS_KEY.FINISHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.SIGNIFICANCE_2_FINISHED:
            origin_status = JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL
            target_status = JOB_STATUS_KEY.FINISHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.ALLSCHED_2_FINISHED:
            origin_status = JOB_STATUS_KEY.DONE_ALL_SCHED
            target_status = JOB_STATUS_KEY.FINISHED        
        elif status_update_path == JOB_STATUS_UPDATE_PATH.RUNNING_2_FINISHED:
            origin_status = JOB_STATUS_KEY.RUNNING
            target_status = JOB_STATUS_KEY.FINISHED
        elif status_update_path == JOB_STATUS_UPDATE_PATH.SIMULATION_NOSUBMIT_2_NOSHCED:
            origin_status = JOB_STATUS_KEY.SIMULATION_NO_SUMBIT
            target_status = JOB_STATUS_KEY.NO_SCHE
        else:
            raise ValueError("status_update_path: {} => origin_status: {}; target_status: {}".format(
                status_update_path, origin_status, target_status
            ))
        return origin_status, target_status

    def get_job_datablock_significance_sync(self, type_id, all_significance_state):
        sub_train_index_2_significance = {
            index: None for index, _ in enumerate(all_significance_state)
        }
        significance_list = self.significance_policy.get_job_significance_result_for_all_datablocks(all_significance_state)
        for index, signi in enumerate(significance_list):
            sub_train_index_2_significance[index] = signi
        # 将所有为None的提取出来, 异步请求一下
        async_indexes = [k for k, v in sub_train_index_2_significance.items() if v is None]
        if len(async_indexes) > 0:
            self.sched_logger.warning("enter into get_job_datablock_significance_async!")
            device_list = [0, 1, 2, 3] * 15 # TODO(xlc): 直接用sched处理即可
            async_significance_state = []
            for index in async_indexes:
                async_significance_state.append(all_significance_state[index])

            async_significance_list = self.significance_policy.get_job_datablock_significance_async(async_significance_state, device_list)
            for index, signi in enumerate(async_significance_list):
                sub_train_index_2_significance[async_indexes[index]] = signi
        # 最终返回: {sub_train_key_id: value}
        sub_train_datasetidentifier_2_significance = {
            all_significance_state[key]["sub_train_key_id"]: value for key, value in sub_train_index_2_significance.items()
        }
        return sub_train_datasetidentifier_2_significance
    
    '''
    def sched_dispatch_testbed_start(self, cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_times):
        self.sched_logger.info("")
        def thread_func_sched_dispatch_testbed(cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_times):
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.calculate_significance_for_nosched_jobs()
                if cal_significance_sleep_time > 0:
                    time.sleep(cal_significance_sleep_time)
                self.sched_dataset_for_done_significance_cal_jobs()
                if scheduler_update_sleep_time > 0:
                    time.sleep(scheduler_update_sleep_time)
                self.placement_dispatch_for_allsched_jobs()
                if placement_sleep_times > 0:
                    time.sleep(placement_sleep_times)
            self.sched_logger.info("Thread [thread_func_sched_dispatch_testbed] finished!")
        p = threading.Thread(target=thread_func_sched_dispatch_testbed, args=(cal_significance_sleep_time, scheduler_update_sleep_time, placement_sleep_times), daemon=True)
        self.all_testbed_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_sched_dispatch_testbed] started!")
    
                
    def real_recoming_jobs(self):
        all_recoming_jobs_copy = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.RECOMING])
        if len(all_recoming_jobs_copy) <= 0:
            return 
        for job_id in all_recoming_jobs_copy:
            current_time = time.time()
            if job_id not in self.jobid_2_recoming_min_time:
                self.sched_logger.warning("job_id [{}] not in self.jobid_2_recoming_min_time".format(job_id))
                self.jobid_2_recoming_min_time[job_id] = current_time + self.recoming_time_interval
                continue
            if self.jobid_2_recoming_min_time[job_id] <= current_time:
                status_update_path, _ = self.get_target_job_status_update_path_and_status(job_id, "real_recoming")
                origin_status_success, target_status_success = self.get_job_status_update_origin_target(status_update_path)
                self.sche_reflash_job_status(job_id, origin_status_success, target_status_success)
                self.sched_logger.info("job_id [{}] recoming".format(job_id))
    '''
                
    def calculate_significance_for_nosched_jobs(self):
        # self.sched_logger.debug("in calculate_significance_for_nosched_jobs")
        all_no_sche_jobs = self.status_2_jobid[JOB_STATUS_KEY.NO_SCHE]
        if len(all_no_sche_jobs) <= 0:
            return 
        for job_id in all_no_sche_jobs:
            if job_id in self.jobid_2_target_significance and self.jobid_2_target_significance[job_id] is None:
                continue
            test_dataset_name = self.jobid_2_test_dataset_name[job_id]
            sub_test_key_id = self.jobid_2_sub_test_key_id[job_id]
            model_name = self.jobid_2_origininfo[job_id]["model_name"]

            job_target_train_dataset_name = self.jobid_2_train_dataset_name[job_id]
            # self.sched_logger.debug(f"job[{job_id}] target_train_dataset_name: {job_target_train_dataset_name}")
            # self.sched_logger.debug(f"sub_train_datasetidentifier_2_dataset_status: {self.sub_train_datasetidentifier_2_dataset_status}")
            
            if job_target_train_dataset_name in self.sub_train_datasetidentifier_2_dataset_status:
                all_significance_state = []
                for sub_train_key_id in self.sub_train_datasetidentifier_2_dataset_status[job_target_train_dataset_name]:
                    significance_state = {
                        "test_dataset_name": test_dataset_name,
                        "sub_test_key_id": sub_test_key_id,
                        "train_dataset_name": job_target_train_dataset_name,
                        "sub_train_key_id": sub_train_key_id,
                        "model_name": model_name
                    }
                    all_significance_state.append(significance_state)
            else:
                all_significance_state = []
            type_id = self.jobid_2_typeid[job_id]
            self.jobid_2_target_significance[job_id] = self.get_job_datablock_significance_sync(type_id, all_significance_state)
            
            if min(self.jobid_2_target_significance[job_id].values()) / self.jobid_2_target_siton_run_epsilon[job_id] < self.min_significance_epsilon_ratio:
                self.min_significance_epsilon_ratio = min(self.jobid_2_target_significance[job_id].values()) / self.jobid_2_target_siton_run_epsilon[job_id]
            if max(self.jobid_2_target_significance[job_id].values()) / self.jobid_2_target_siton_run_epsilon[job_id] > self.max_significance_epsilon_ratio:
                self.max_significance_epsilon_ratio = max(self.jobid_2_target_significance[job_id].values()) / self.jobid_2_target_siton_run_epsilon[job_id]
            self.sched_logger.info(f"min_significance_epsilon_ratio(L): {self.min_significance_epsilon_ratio}")
            self.sched_logger.info(f"max_significance_epsilon_ratio(U): {self.max_significance_epsilon_ratio}")
            # self.jobid_2_real_significance[job_id].append(copy.deepcopy(self.jobid_2_target_significance[job_id]))
            self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.NO_SCHE, JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL)

    def sched_dataset_for_jobs(self, sched_job_origin_state):
        assert sched_job_origin_state == JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL or sched_job_origin_state == JOB_STATUS_KEY.WAITING
        # self.sched_logger.debug("in sched_dataset_for_jobs")
        need_operator_jobs = copy.deepcopy(self.status_2_jobid[sched_job_origin_state])
        # self.sched_logger.debug(f"need_operator_jobs first: {need_operator_jobs}")
        if sched_job_origin_state == JOB_STATUS_KEY.WAITING:
            need_operator_jobs = [job_id for job_id in need_operator_jobs if self.jobid_2_need_instantly_recoming[job_id]]
            for job_id in need_operator_jobs:
                if self.simulation:
                    self.jobid_2_arrival_time[job_id].append(self.simulation_global_time)
                else:
                    self.jobid_2_arrival_time[job_id].append(self.testbed_current_time)
                self.jobid_2_need_instantly_recoming[job_id] = False
        if len(need_operator_jobs) <= 0:
            return
        self.sched_logger.debug(f"sched_dataset_for_jobs in {sched_job_origin_state} len(need_operator_jobs): {len(need_operator_jobs)}")
        # self.sched_logger.debug(f"need_operator_jobs second: {need_operator_jobs}")
        if self.assignment_policy.only_one: # 这里其实应该改成一个for循环, 一次一次来, 直到当前所有的新来的任务完成决策
            for operator_job in need_operator_jobs:
                current_operator_jobs = [operator_job]
                self.sched_dataset_for_current_operator_jobs(current_operator_jobs, sched_job_origin_state)
        else:
            self.sched_dataset_for_current_operator_jobs(need_operator_jobs, sched_job_origin_state)
        
    def sched_dataset_for_current_operator_jobs(self, current_operator_jobs, sched_job_origin_state):
        job_id_2_dataset_name = {}
        job_id_2_target_epsilon_require = {}
        job_id_2_target_datablock_selected_num = {}
        job_id_2_job_priority_weight = {}
        job_id_2_test_dataset_name = {}
        job_id_2_sub_test_key_id = {}
        job_id_2_target_significance = {}
        job_id_2_arrival_index = {}
        job_id_2_arrival_time = {}
        # self.sched_logger.debug(f"current_operator_jobs: {current_operator_jobs}")
        for job_id in current_operator_jobs:
            job_current_siton_run_index = self.jobid_2_current_operate_siton_run_index[job_id]
            job_id_2_dataset_name[job_id] = self.jobid_2_train_dataset_name[job_id]
            job_id_2_target_epsilon_require[job_id] = self.jobid_2_target_siton_run_epsilon[job_id]
            job_id_2_target_datablock_selected_num[job_id] = self.jobid_2_datablock_selected_num[job_id]
            job_id_2_job_priority_weight[job_id] = self.jobid_2_priority_weight[job_id]
            job_id_2_test_dataset_name[job_id] = self.jobid_2_test_dataset_name[job_id]
            job_id_2_sub_test_key_id[job_id] = self.jobid_2_sub_test_key_id[job_id]
            job_id_2_target_significance[job_id] = self.jobid_2_target_significance[job_id]
            job_id_2_arrival_index[job_id] = self.jobid_2_arrival_index[job_id][job_current_siton_run_index]
            job_id_2_arrival_time[job_id] = self.jobid_2_arrival_time[job_id][job_current_siton_run_index]

        # 为没有决定分配方案的任务决定分配方案
        job_2_selected_datablock_identifiers, waiting_job_ids, \
            selected_real_sched_epsilon_map, calcu_compare_epsilon, job_2_instant_recoming_flag = \
            self.get_scheduling_datablock_result_from_policy(job_id_2_dataset_name, 
                job_id_2_target_epsilon_require, job_id_2_target_datablock_selected_num, job_id_2_job_priority_weight, 
                job_id_2_test_dataset_name, job_id_2_sub_test_key_id, 
                job_id_2_target_significance, job_id_2_arrival_index, 
                job_id_2_arrival_time)

        instant_recoming_flag_num = 0
        for temp_job_id, instant_recoming_flag in job_2_instant_recoming_flag.items():
            if instant_recoming_flag:
                instant_recoming_flag_num += 1
                self.jobid_2_need_instantly_recoming[temp_job_id] = True
        if self.simulation and instant_recoming_flag_num > 0:
            self.simulation_queue.put(SchedEvent(self.simulation_global_time, EVENT_KEY.JOB_SIGCAL_SCHED_RUNNING, {}))
        
        success_datasetidentifier_2_consume_epsilon = {}
        success_sched_job_ids = set()
        if len(job_2_selected_datablock_identifiers) > 0:
            temp_copy_sub_train_datasetidentifier_2_epsilon_remain = copy.deepcopy(
                self.sub_train_datasetidentifier_2_epsilon_remain
            )
            for temp_job_id, selected_datablock_identifiers in job_2_selected_datablock_identifiers.items():
                dataset_name = job_id_2_dataset_name[temp_job_id]
                if self.all_or_nothing_flag and len(selected_datablock_identifiers) != job_id_2_target_datablock_selected_num[temp_job_id]:
                    continue
                job_temp_sched_success_flag = True
                for identifier in selected_datablock_identifiers:
                    consume_epsilon = selected_real_sched_epsilon_map[(temp_job_id, identifier)]
                    if consume_epsilon > temp_copy_sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier]:
                        job_temp_sched_success_flag = False
                        break
                    else:
                        temp_copy_sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] -= consume_epsilon
                if job_temp_sched_success_flag:
                    success_sched_job_ids.add(temp_job_id)
                else:
                    self.sched_logger.warning(f"it's wrong! temp_job_id[{temp_job_id}] has not choice consume: {consume_epsilon}; capacity: {temp_copy_sub_train_datasetidentifier_2_epsilon_remain}")
                    continue
                
            # 实质调度
            for temp_job_id in success_sched_job_ids:
                dataset_name = job_id_2_dataset_name[temp_job_id]
                selected_datablock_identifiers = job_2_selected_datablock_identifiers[temp_job_id]
                job_current_siton_run_index = self.jobid_2_current_operate_siton_run_index[temp_job_id]
                self.jobid_2_real_significance[temp_job_id].append(copy.deepcopy(self.jobid_2_target_significance[job_id])) # 只有成功的时候才会被调用, 因此还是直接修改后push进去算了
                for identifier in selected_datablock_identifiers:
                    consume_epsilon = selected_real_sched_epsilon_map[(temp_job_id, identifier)] 
                    self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] -= consume_epsilon
                    if self.sub_train_datasetidentifier_2_epsilon_remain[dataset_name][identifier] <= 0.0:
                        self.sub_train_datasetidentifier_2_dataset_status[dataset_name][identifier] = DATASET_STATUS_KEY.EXHAUST
                        if self.simulation:
                            self.sub_train_datasetidentifier_2_exhausted_time[dataset_name][identifier] = self.simulation_global_time
                        else:
                            self.sub_train_datasetidentifier_2_exhausted_time[dataset_name][identifier] = self.testbed_current_time
                    if dataset_name not in success_datasetidentifier_2_consume_epsilon:
                        success_datasetidentifier_2_consume_epsilon[dataset_name] = {}
                    if identifier not in success_datasetidentifier_2_consume_epsilon[dataset_name]:
                        success_datasetidentifier_2_consume_epsilon[dataset_name][identifier] = 0.0
                    success_datasetidentifier_2_consume_epsilon[dataset_name][identifier] += consume_epsilon
                    self.jobid_2_sched_siton_run_epsilon[temp_job_id] = consume_epsilon
                    self.jobid_2_real_significance[temp_job_id][-1][identifier] = (self.jobid_2_sched_siton_run_epsilon[temp_job_id] / self.jobid_2_target_siton_run_epsilon[temp_job_id]) * self.jobid_2_target_significance[temp_job_id][identifier]
                    
                if temp_job_id not in self.jobid_2_sub_train_key_ids:
                    self.jobid_2_sub_train_key_ids[temp_job_id] = []
                self.jobid_2_sub_train_key_ids[temp_job_id].append(selected_datablock_identifiers)
                
        
        self.sched_logger.info("final true success Jobs selected datablock identifiers: {}".format(success_sched_job_ids))
        self.push_success_scheduling_result_to_policy(success_datasetidentifier_2_consume_epsilon)

        # 不管是否进行数据块的分配, 都应该增加历史记录, 这样可以在前期的历史记录收集阶段做一些优化! 
        # TODO(xlc): 可能会导致历史记录的数量特别多!
        for temp_job_id in current_operator_jobs:
            self.job_add_to_history(temp_job_id)

        need_failed_job = copy.deepcopy(current_operator_jobs)
        for temp_job_id in success_sched_job_ids:
            self.sched_logger.debug(f"final true success jobid_2_sub_train_key_ids [{temp_job_id}] <=> [{self.jobid_2_sub_train_key_ids[temp_job_id][-1]}]")
            need_failed_job.remove(temp_job_id)
            status_update_path, _ = self.get_target_job_status_update_path_and_status(temp_job_id, "dataset")
            origin_status_success, target_status_success = self.get_job_status_update_origin_target(status_update_path)
            self.sche_reflash_job_status(temp_job_id, origin_status_success, target_status_success)

        # self.sched_logger.debug(f"waiting_job_ids: {waiting_job_ids}")
        for temp_job_id in waiting_job_ids:
            need_failed_job.remove(temp_job_id)
            self.worker_waiting_job_callback(temp_job_id, self.jobid_2_origininfo[temp_job_id])
        
        # self.sched_logger.debug(f"need_failed_job: {need_failed_job}")
        for temp_job_id in need_failed_job:
            self.sched_logger.info("failed job scheduling [{}] first".format(temp_job_id))
            # TODO(xlc): 这里不应该直接设置为Failed状态, 而是考虑max_time的情况, 决定是否将任务放到NO_SCHED的状态, 同时需要知道模型的最新保存位置?
            self.worker_failed_job_callback(temp_job_id, self.jobid_2_origininfo[temp_job_id], FAILED_RESULT_KEY.SCHED_FAILED)

        self.sched_logger.debug(f"waiting job[{len(self.status_2_jobid[JOB_STATUS_KEY.WAITING])}]: {self.status_2_jobid[JOB_STATUS_KEY.WAITING]}")
        self.sched_logger.debug(f"sub_train_datasetidentifier_2_epsilon_remain: {self.sub_train_datasetidentifier_2_epsilon_remain}")
        # if not self.simulation:
            # self.report_status("after sched_dataset_for_done_significance_cal_jobs")

    def placement_dispatch_for_allsched_jobs(self):
        # self.sched_logger.debug("in placement_dispatch_for_allsched_jobs")
        # 放置任务
        all_done_all_sched_jobs_copy = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.DONE_ALL_SCHED])
        if len(all_done_all_sched_jobs_copy) <= 0:
            return 
        args = []
        for job_id in all_done_all_sched_jobs_copy:
            gpuidentifier_enable_status = [k for k, v in self.gpuidentifier_2_jobinstances.items() if len(v) < self.max_gpu_fuzai]
            if len(gpuidentifier_enable_status) > 0:
                current_siton_run_index = self.jobid_2_current_operate_siton_run_index[job_id]
                gpu_identifer = random.choice(gpuidentifier_enable_status)
                self.gpuidentifier_2_jobinstances[gpu_identifer].append(job_id) 
                self.jobid_2_gputarget[job_id] = gpu_identifer
                worker_ip, worker_gpu_id = self.get_worker_identifier_detail(gpu_identifer)
                worker_port = self.workerip_2_ports[worker_ip]

                origin_info = self.jobid_2_origininfo[job_id]
                sched_epsilon_one_siton_run = self.jobid_2_sched_siton_run_epsilon[job_id]
                worker_dataset_config = {
                    "train_dataset_name": self.jobid_2_train_dataset_name[job_id],
                    "test_dataset_name": self.jobid_2_test_dataset_name[job_id],
                    "sub_train_key_ids": self.jobid_2_sub_train_key_ids[job_id][-1],
                    "sub_test_key_id": self.jobid_2_sub_test_key_id[job_id],
                    "sub_train_dataset_config_path": os.path.join(DATASET_PATH, self.dataset_name, f"{self.dataset_config_name}.json"),
                    "test_dataset_config_path": os.path.join(DATASET_PATH, self.dataset_name, f"subtest.json")
                }
                model_save_path = self.jobid_2_model_save_path[job_id]
                logging_file_path = self.jobid_2_logging_file_path[job_id]
                summary_writer_path = self.summary_writer_path
                summary_writer_key = self.jobid_2_summary_writer_key[job_id]
                siton_run_epoch_num = self.jobid_2_siton_run_epoch_num[job_id]
                current_success_siton_run_num = self.jobid_2_success_siton_run_num[job_id]
                begin_epoch_num = current_success_siton_run_num * siton_run_epoch_num
                current_block_selected_num = len(self.jobid_2_sub_train_key_ids[job_id][-1])

                final_significance = 0.0
                for sub_train_key_id in self.jobid_2_sub_train_key_ids[job_id][-1]:
                    final_significance += self.jobid_2_real_significance[job_id][-1][sub_train_key_id]
                
                if self.simulation:
                    self.jobid_2_started_time[job_id].append(self.simulation_global_time)
                else:
                    self.jobid_2_started_time[job_id].append(self.testbed_current_time)
                
                self.sche_reflash_job_status(job_id, JOB_STATUS_KEY.DONE_ALL_SCHED, JOB_STATUS_KEY.RUNNING)
                if not self.simulation:
                    args.append([job_id, origin_info, sched_epsilon_one_siton_run, siton_run_epoch_num, begin_epoch_num, worker_ip, worker_port, worker_gpu_id, 
                                worker_dataset_config, model_save_path, summary_writer_path, summary_writer_key, logging_file_path, final_significance, self.simulation])
                else:
                    if len(self.jobid_2_results[job_id]) > 0:
                        last_result = self.jobid_2_results[job_id][-1]
                        new_test_acc = last_result['test_acc'] + origin_info['siton_up_test_acc'] * (final_significance / current_block_selected_num)
                    else:
                        last_result_acc = origin_info['simulation_init_test_acc']
                        new_test_acc = last_result_acc + origin_info['siton_up_test_acc'] * (final_significance / current_block_selected_num)
                    all_results = {
                        'train_acc': 0.0,
                        'train_loss': 0.0,
                        'test_acc': new_test_acc,
                        'test_loss': 0.0,
                        'epsilon_consume': sched_epsilon_one_siton_run,
                        'begin_epoch_num': begin_epoch_num,
                        'run_epoch_num': siton_run_epoch_num,
                        'final_significance': final_significance
                    }
                    self.worker_finished_job_callback(job_id, origin_info, all_results)
        if not self.simulation and len(args) > 0:
                # 转置
            final_args = [[row[i] for row in args] for i in range(len(args[0]))]
            with ThreadPoolExecutor(max_workers=len(args)) as pool:
                pool.map(DL_server_do_jobs, *final_args)
         
                

    '''
    def sched_best_serve_for_failed_jobs(self):
        all_failed_sched_jobs_copy = copy.deepcopy(self.status_2_jobid[JOB_STATUS_KEY.FAILED])
        if len(all_failed_sched_jobs_copy) <= 0:
            return 
        # 注意, 这里只会执行一次, 完成For循环后就要将任务的状态改变, 能上就该状态为DONE_ALL_SCHDE，不能上则保留为Failed，然后传输给dispatcher
        # TODO(xlc): 需要进行一波修改, 但是比较麻烦
        for job_id in all_failed_sched_jobs_copy:
            self.sched_logger.info("failed job scheduling [{}] final".format(job_id))
            origin_info = self.jobid_2_origininfo[job_id]
            self.failed_job_to_dispatcher(job_id, origin_info, is_first_failed=False)

    def set_final_job_flag(self, status):
        self.final_job_come = status
        self.sched_logger.info("set_final_job_flag: {}".format(status))

    def sched_best_serve_start(self, best_serve_time):
        def thread_func_timely_best_serve(best_serve_time):
            while not self.all_finished and self.final_job_come and self.finished_update_init_history_jobs:
                self.sched_best_serve_for_failed_jobs()
                time.sleep(best_serve_time)
            self.sched_logger.info("Thread [thread_func_timely_best_serve] finished!")
        p = threading.Thread(target=thread_func_timely_best_serve, args=(best_serve_time, ), daemon=True)
        self.best_serve_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_timely_best_serve] started!")
    '''

    def sched_dispatch_start(self, scheduler_update_sleep_time):
        def thread_func_timely_schedule(scheduler_update_sleep_time):
            while (not self.all_finished) and self.finished_update_init_history_jobs:
                self.sched_dataset_for_jobs(JOB_STATUS_KEY.DONE_SIGNIFICANCE_CAL)
                if self.enable_waiting_flag:
                    self.sched_dataset_for_jobs(JOB_STATUS_KEY.WAITING)
                if not self.simulation:
                    time.sleep(scheduler_update_sleep_time)
            self.sched_logger.info("Thread [thread_func_timely_schedule] finished!")
        p = threading.Thread(target=thread_func_timely_schedule, args=(scheduler_update_sleep_time, ), daemon=True)
        self.sched_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_timely_schedule] started!")

    def cal_significance_dispatch_start(self, cal_significance_sleep_time):
        def thread_func_timely_cal_significance(cal_significance_sleep_time):
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.calculate_significance_for_nosched_jobs()
                if not self.simulation:
                    time.sleep(cal_significance_sleep_time)
            self.sched_logger.info("Thread [thread_func_timely_cal_significance] finished!")
        p = threading.Thread(target=thread_func_timely_cal_significance, args=(cal_significance_sleep_time, ), daemon=True)
        self.cal_significance_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_timely_cal_significance] started!")

    def placement_dispatch_start(self, placement_sleep_time):
        def thread_func_timely_placement(placement_sleep_time):
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.placement_dispatch_for_allsched_jobs()
                if not self.simulation:
                    time.sleep(placement_sleep_time)
            self.sched_logger.info("Thread [thread_func_timely_placement] finished!")
        p = threading.Thread(target=thread_func_timely_placement, args=(placement_sleep_time, ), daemon=True)
        self.placement_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_timely_placement] started!")

    def sched_update_current_time(self, testbed_start_time):
        def thread_func_timely_update_time():
            while (not self.all_finished) and (not self.all_stop):
                self.testbed_current_time = time.time() - self.testbed_start_time
                # self.sched_logger.debug(f"sched testbed_current_time: {self.testbed_current_time}")
                time.sleep(1)
            self.sched_logger.info("Thread [thread_func_timely_update_time] finished!")
        self.testbed_start_time = testbed_start_time
        p = threading.Thread(target=thread_func_timely_update_time, daemon=True)
        p.start()
        self.sched_logger.info("Thread [thread_func_timely_update_time] start!")

    '''
    def recoming_jobs_start(self, real_coming_sleep_time):
        def thread_func_real_for_recoming_jobs(real_coming_sleep_time):
            while not self.all_finished and self.finished_update_init_history_jobs:
                self.real_recoming_jobs()
                time.sleep(real_coming_sleep_time)
            self.sched_logger.info("Thread [thread_func_real_for_recoming_jobs] finished!")
        p = threading.Thread(target=thread_func_real_for_recoming_jobs, args=(real_coming_sleep_time, ), daemon=True)
        self.real_recoming_thread = p
        p.start()
        self.sched_logger.info("Thread [thread_func_real_for_recoming_jobs] started!")
    '''

    def restart_sched(self):
        self.all_finished = False

    def sched_end(self):
        self.all_finished = True
        self.all_testbed_thread = None
        self.sched_thread = None
        self.cal_significance_thread = None
        self.placement_thread = None
        self.gpu_thread = None

        self.finished_job_to_dispatcher_list = []
        self.finished_job_to_dispatcher_thread = None
        self.failed_job_to_dispatcher_list = []
        self.failed_job_to_dispatcher_thread = None

    def end_and_report_dispatchers_by_sched(self):
        current_success_num, current_failed_num, current_no_submit_num, current_no_sche_num, \
            current_done_sig_num, current_done_sche_num, current_running_num, current_recoming_num = \
            self.get_all_state_num()

        all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_final_significance = self.get_result_status()
        
        all_target_datablock_num = 0
        all_success_datablock_num = 0
        for _, target_selected_num in self.jobid_2_datablock_selected_num.items():
            all_target_datablock_num += target_selected_num
        for _, success_selected_datablocks in self.jobid_2_sub_train_key_ids.items():
            temp_one_job_success_selected_datablock_num = 0
            for once_selected_datablock in success_selected_datablocks:
                temp_one_job_success_selected_datablock_num += len(once_selected_datablock)
            if len(success_selected_datablocks) > 0:
                all_success_datablock_num += (temp_one_job_success_selected_datablock_num / len(success_selected_datablocks))
        all_failed_datablock_num = all_target_datablock_num - all_success_datablock_num

        self.sched_logger.info(f"all_target_datablock_num: {all_target_datablock_num}")
        self.sched_logger.info(f"all_success_datablock_num: {all_success_datablock_num}")
        self.sched_logger.info(f"all_failed_datablock_num: {all_failed_datablock_num}")

        result_map = {
            "current_success_num": current_success_num,
            "current_failed_num": current_failed_num,
            "current_no_submit_num": current_no_submit_num,
            "current_no_sche_num": current_no_sche_num,
            "current_done_sig_num": current_done_sig_num,
            "current_done_sche_num": current_done_sche_num,
            "current_running_num": current_running_num,
            "current_recoming_num": current_recoming_num,
            "all_train_loss": all_train_loss,
            "all_train_accuracy": all_train_accuracy,
            "all_test_loss": all_test_loss,
            "all_test_accuracy": all_test_accuracy,
            "all_final_significance": all_final_significance,
            "all_target_datablock_num": all_target_datablock_num,
            "all_success_datablock_num": all_success_datablock_num,
            "all_failed_datablock_num": all_failed_datablock_num
        }

        return result_map

    def sched_update_gpu_status_start(self, init_workerip_2_ports, init_gpuidentifiers, current_test_all_dir, simulation_index):
        self.sched_logger.info("Thread [thread_func_timely_update_gpu] started!")
        for worker_ip, worker_port in init_workerip_2_ports.items():
            self.workerip_2_ports[worker_ip] = worker_port
            work_client = zerorpc.Client()
            work_client.connect("tcp://{}:{}".format(worker_ip, worker_port))
            work_client.initialize_logging_path(current_test_all_dir, simulation_index)
        for gpu_identifier in init_gpuidentifiers:
            self.gpuidentifier_2_jobinstances[gpu_identifier] = []
        self.sched_logger.info("Thread [thread_func_timely_update_gpu] success!")

        
    def sched_update_assignment_policy(self, assignment_policy, assignment_args):
        if assignment_policy == "PBGPolicy" or assignment_policy == "PBG":
            pipeline_sequence_all_num, job_request_all_num, comparison_cost_epsilon, comparison_z_threshold, L, U = assignment_args
            policy_item = PBGPolicy(pipeline_sequence_all_num, job_request_all_num, comparison_cost_epsilon, comparison_z_threshold, L, U, self.seed, self.sched_logger)
        elif assignment_policy == "PBGMixPolicy" or assignment_policy == "PBGMix":
            pipeline_sequence_all_num, job_request_all_num, comparison_cost_epsilon, comparison_z_threshold, L, U, gitta = assignment_args
            policy_item = PBGMixPolicy(pipeline_sequence_all_num, job_request_all_num, comparison_cost_epsilon, comparison_z_threshold, L, U, gitta, self.seed, self.sched_logger)
        elif assignment_policy == "HISwithOrderProVersionPolicy" or assignment_policy == "HISwithOrderProVersion":
            beta, pipeline_sequence_all_num, job_request_all_num, infinity_flag = assignment_args
            policy_item = HISwithOrderProVersionPolicy(beta, pipeline_sequence_all_num, job_request_all_num, infinity_flag, self.seed, self.sched_logger)
        elif assignment_policy == "IterativeHISwithOrderProVersionPolicy" or assignment_policy == "IterativeHISwithOrderProVersion":
            beta, pipeline_sequence_all_num, job_request_all_num, batch_size_for_one_epoch, infinity_flag = assignment_args
            policy_item = IterativeHISwithOrderProVersionPolicy(beta, pipeline_sequence_all_num, job_request_all_num, batch_size_for_one_epoch, infinity_flag, self.seed, self.sched_logger)
        elif assignment_policy == "SagewithRemainPolicy" or assignment_policy == "SagewithRemain":
            pipeline_sequence_all_num, job_request_all_num = assignment_args
            policy_item = SagewithRemainPolicy(pipeline_sequence_all_num, job_request_all_num, self.seed, self.sched_logger)
        elif assignment_policy == "BestFitwithRemainPolicy" or assignment_policy == "BestFitwithRemain":
            pipeline_sequence_all_num, job_request_all_num = assignment_args
            policy_item = BestFitwithRemainPolicy(pipeline_sequence_all_num, job_request_all_num, self.seed, self.sched_logger)
        elif assignment_policy == "OfflinePolicy" or assignment_policy == "Offline":
            pipeline_sequence_all_num, job_request_all_num = assignment_args
            policy_item = OfflinePolicy(pipeline_sequence_all_num, job_request_all_num, self.seed, self.sched_logger)
        self.assignment_policy = policy_item
        self.assignment_policy.report_state()

    def sched_update_significance_policy(self, significance_policy, significance_args):
        if significance_policy == "HISOTDDPolicy":
            raise ValueError(f"assignment_policy: {significance_policy} is abandoned!")
            self.significance_policy = HISOTDDPolicy(self.simulation, self.sched_logger)
        elif significance_policy == "OTDDPolicy":
            self.significance_policy = OTDDPolicy(self.dataset_name, self.dataset_config_name, self.simulation, self.sched_logger)
        elif significance_policy == "HVOTDDPolicy":
            raise ValueError(f"assignment_policy: {significance_policy} is abandoned!")
            self.significance_policy = HVOTDDPolicy(self.simulation, self.sched_logger)
        elif significance_policy == "HVPolicy":
            raise ValueError(f"assignment_policy: {significance_policy} is abandoned!")
            self.significance_policy = HVPolicy(self.simulation, self.sched_logger)
        elif significance_policy == "TempPolicy":
            metric = significance_args
            self.significance_policy = TempPolicy(self.dataset_name, self.dataset_config_name, metric, self.simulation, self.sched_logger)
        self.sched_logger.info("significance_policy: {}".format(self.significance_policy.name))
        
def scheduler_listener_func(scheduler_server_item):
    # def sched_func_timely(scheduler_server_item):
    #     s = zerorpc.Server(scheduler_server_item)
    #     ip_port = "tcp://0.0.0.0:{}".format(scheduler_server_item.sched_port)
    #     s.bind(ip_port)
    #     print("DL_server running in {}".format(ip_port))
    #     s.run()
    #     print("self.sched_logger.info sth...")
    # p = threading.Thread(target=sched_func_timely, args=(scheduler_server_item, ), daemon=True)
    # p.start()
    s = zerorpc.Server(scheduler_server_item)
    ip_port = "tcp://0.0.0.0:{}".format(scheduler_server_item.sched_port)
    s.bind(ip_port)
    print("DL_server running in {}".format(ip_port))
    g = zerorpc.gevent.spawn(s.run)
    return g


def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--worker_ips", type=str, nargs="+", default=["172.18.162.2", "172.18.162.3", "172.18.162.4", "172.18.162.5", "172.18.162.6"])
    parser.add_argument("--worker_ports", type=int, nargs="+", default=[16202, 16203, 16204, 16205, 16206])
    
    parser.add_argument("--sched_ip", type=str, default="172.18.162.6")
    parser.add_argument("--sched_port", type=int, default=16306)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_df_config()

    sched_ip = args.sched_ip
    sched_port = args.sched_port

    scheduler_server_item = Scheduler_server(sched_ip, sched_port)
    sched_p = scheduler_listener_func(scheduler_server_item)

    while not scheduler_server_item.all_stop:
        zerorpc.gevent.sleep(10)
    print("DL sched finished!!")
    
    sys.exit(0)