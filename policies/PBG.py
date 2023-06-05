from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
import json

class PBGPolicy(Policy):
    def __init__(self, pipeline_sequence_all_num, job_request_all_num, comparison_cost_epsilon, comparison_z_threshold, L, U, seed, logger):
        super().__init__(pipeline_sequence_all_num, job_request_all_num)
        self._name = 'PBGPolicy'
        self.comparison_cost_epsilon = comparison_cost_epsilon
        self.comparison_z_threshold = comparison_z_threshold
        self.L = L
        self.U = U
        self.logger = logger
        self.waiting_queue_capacity = 1

        self.only_one = True
        self.need_history = False
        self.initialize_seeds(seed)

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: comparison_cost_epsilon: {}".format(self.comparison_cost_epsilon))
        self.logger.info("policy args: comparison_z_threshold: {}".format(self.comparison_z_threshold))
        self.logger.info("policy args: L: {}".format(self.L))
        self.logger.info("policy args: U: {}".format(self.U))

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

    def Lap(self, scale):
        return np.random.laplace(loc=0.0, scale=scale)

    def Threshold_func(self, datablock_z):
        if 0.0 <= datablock_z <= self.comparison_z_threshold:
            return self.L
        elif self.comparison_z_threshold < datablock_z <= 1.0:
            return math.pow(self.U * math.exp(1) / self.L, datablock_z) * self.L / math.exp(1)
        return self.L

    def filter_by_threshold(self, datablock_epsilon_capacity, datablock_z, target_epsilon_consume, significance_plus_weight):
        if datablock_z < self.comparison_z_threshold:
            is_select = True
            # new_z = datablock_z + target_epsilon_consume / datablock_epsilon_capacity
            compare_epsilon = 0.0
        else:
            if self.comparison_cost_epsilon > 0.0:
                if ((significance_plus_weight / (target_epsilon_consume + self.comparison_cost_epsilon)) + self.Lap(4.0/self.comparison_cost_epsilon) > self.Threshold_func(datablock_z) + self.Lap(2.0/self.comparison_cost_epsilon) 
                and target_epsilon_consume + self.comparison_cost_epsilon < (1.0 - datablock_z) * datablock_epsilon_capacity):
                    is_select = True
                    compare_epsilon = self.comparison_cost_epsilon
                
                else:
                    is_select = False
                    compare_epsilon = 0.0
            else:
                if ((significance_plus_weight / target_epsilon_consume) > self.Threshold_func(datablock_z)
                and target_epsilon_consume < (1.0 - datablock_z) * datablock_epsilon_capacity):
                    is_select = True
                    compare_epsilon = 0.0
                else:
                    is_select = False
                    compare_epsilon = 0.0
        return is_select, compare_epsilon
        
    def get_allocation(self, state, all_or_nothing_flag, enable_waiting_flag):
        job_id, train_dataset_name = self.get_allocation_judge_one_job(state)
        self.add_to_policy_profiler(job_id)
        
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
        target_epsilon_require = state["job_id_2_target_epsilon_require"][job_id]
        target_datablock_select_num = state["job_id_2_target_datablock_selected_num"][job_id]
        job_priority_weight = state["job_id_2_job_priority_weight"][job_id]
        sub_train_datasetidentifier_2_significance = state["job_id_2_significance"][job_id]

        temp_datasetidentifier_2_epsilon_z = {
            datasetidentifier: 1.0-sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier]/sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain
        }
        count = 0
        temp_selected_datablock_identifiers = []
        temp_selected_real_sched_epsilon_map = {}
        calcu_compare_epsilon = 0.0
        
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取随机一个数据集
            datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]
            significance_plus_weight = job_priority_weight * sub_train_datasetidentifier_2_significance[datasetidentifier]
            
            is_select, compare_epsilon = self.filter_by_threshold(datablock_epsilon_capacity, datablock_z, target_epsilon_require, significance_plus_weight)
            if is_select and sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier] >= target_epsilon_require:
                count += 1
                temp_selected_datablock_identifiers.append(datasetidentifier)
                temp_selected_real_sched_epsilon_map[(job_id, datasetidentifier)] = target_epsilon_require
                calcu_compare_epsilon += compare_epsilon
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
        
        result_job_2_selected_datablock_identifiers = {}
        result_selected_real_sched_epsilon_map = {}
        result_job_2_instant_recoming_flag = {}
        result_waiting_job_ids = []

        temp_sched_failed_flag = False
        if ((not all_or_nothing_flag) and len(temp_selected_datablock_identifiers) > 0) or (all_or_nothing_flag and len(temp_selected_datablock_identifiers) == target_datablock_select_num):
            result_job_2_selected_datablock_identifiers[job_id] = temp_selected_datablock_identifiers
            result_selected_real_sched_epsilon_map = temp_selected_real_sched_epsilon_map
        else:
            temp_sched_failed_flag = True

        if enable_waiting_flag: 
            if temp_sched_failed_flag:
                result_waiting_job_ids.append(job_id)
                result_job_2_instant_recoming_flag[job_id] = True
            else:
                result_job_2_instant_recoming_flag[job_id] = True
            
        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name, result_job_2_selected_datablock_identifiers))
        return result_job_2_selected_datablock_identifiers, result_waiting_job_ids, result_selected_real_sched_epsilon_map, calcu_compare_epsilon, result_job_2_instant_recoming_flag