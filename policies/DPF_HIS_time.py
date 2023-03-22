'''
from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math

class WaitingJob(object):
    def __init__(self, target_datablock_select_num, job_arrival_index, target_epsilon_require, select_datablock_alg):
        self.target_datablock_select_num = target_datablock_select_num
        self.job_arrival_index = job_arrival_index
        self.target_epsilon_require = target_epsilon_require
        self.dominant_share = 0.0

        self.select_datablock_alg = select_datablock_alg

class DPFPolicy(Policy):
    def __init__(self, waiting_queue_capacity=10):
        super().__init__()
        self._name = 'DPFPolicy'
        # 保存一个unlocked的量
        self.datablock_identifier_2_unlocked_capacity = {}
        self.waiting_queue = []
        self.waiting_queue_capacity = waiting_queue_capacity

    def get_allocation(self, state):
        target_datablock_select_num = state["target_datablock_select_num"]
        job_arrival_index = state["job_arrival_index"]
        all_job_sequence_num = state["all_job_sequence_num"]
        target_epsilon_require = state["target_epsilon_require"]
        job_id = state["job_id"]

        sub_train_datasetidentifier_2_epsilon_remain = state["all_sub_train_datasetidentifier_2_epsilon_remain"]
        sub_train_datasetidentifier_2_epsilon_capcity = state["all_sub_train_datasetidentifier_2_epsilon_capcity"]
    
        self.on_job_arrival(sub_train_datasetidentifier_2_epsilon_capcity, all_job_sequence_num)
        if len(self.waiting_queue) < self.waiting_queue_capacity:
            waiting_job = WaitingJob(target_datablock_select_num, job_arrival_index, target_epsilon_require)
            self.waiting_queue.append(waiting_job)
        else:
            # 根据dominant_share排序等待任务
            self.on_scheduler_time()

    def on_scheduler_time(self, sub_train_datasetidentifier_2_epsilon_capcity):
        
        for job in self.waiting_queue:
            dominant_share = self.get_dominant_share(
                job.target_epsilon_require,
                sub_train_datasetidentifier_2_epsilon_capcity,   
            )
            job.dominant_share = dominant_share
            
            

    def on_job_arrival(self, sub_train_datasetidentifier_2_epsilon_capcity, all_job_sequence_num):
        for dataset_identifier in sub_train_datasetidentifier_2_epsilon_capcity:
            if dataset_identifier in self.datablock_identifier_2_unlocked_capacity:
                self.datablock_identifier_2_unlocked_capacity[dataset_identifier] = min(
                    sub_train_datasetidentifier_2_epsilon_capcity[dataset_identifier],
                    self.datablock_identifier_2_unlocked_capacity[dataset_identifier] + sub_train_datasetidentifier_2_epsilon_capcity[dataset_identifier] / all_job_sequence_num
                )
            else:
                self.datablock_identifier_2_unlocked_capacity[dataset_identifier] = sub_train_datasetidentifier_2_epsilon_capcity[dataset_identifier] / all_job_sequence_num
        
    def get_dominant_share(self, target_epsilon_require, sub_train_datasetidentifier_2_epsilon_capcity):
        max_dominantshare = 0.0
        result_dataset_identifier = ""
        for dataset_identifier, epsilon_capacity in sub_train_datasetidentifier_2_epsilon_capcity.items():
            if target_epsilon_require / epsilon_capacity > max_dominantshare:
                max_dominantshare = target_epsilon_require / epsilon_capacity
                result_dataset_identifier = dataset_identifier
        assert result_dataset_identifier != ""
        return result_dataset_identifier
        
'''