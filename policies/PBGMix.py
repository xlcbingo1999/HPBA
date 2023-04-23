# 目前看这个算法会对任务的指标造成损失, 可能改动的代码量还不小
# 这个算法不能进行simulation
from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
import json

class PBGMixPolicy(Policy):
    def __init__(self, comparison_cost_epsilon, comparison_z_threshold, L, U, gitta, logger):
        super().__init__()
        self._name = 'PBGMixPolicy'
        self.comparison_cost_epsilon = comparison_cost_epsilon
        self.comparison_z_threshold = comparison_z_threshold
        self.L = L
        self.U = U
        self.gitta = gitta
        self.logger = logger
        self.waiting_queue_capacity = 1

        self.only_one = True
        self.need_history = False

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: comparison_cost_epsilon: {}".format(self.comparison_cost_epsilon))
        self.logger.info("policy args: comparison_z_threshold: {}".format(self.comparison_z_threshold))
        self.logger.info("policy args: L: {}".format(self.L))
        self.logger.info("policy args: U: {}".format(self.U))

    def Lap(self, scale):
        return np.random.laplace(loc=0.0, scale=scale)

    def Threshold_func(self, datablock_z):
        if 0.0 <= datablock_z <= self.comparison_z_threshold:
            return self.L
        elif self.comparison_z_threshold < datablock_z <= 1.0:
            return math.pow(self.U * math.exp(1) / self.L, datablock_z) * self.L / math.exp(1)
        return self.L

    def filter_by_threshold(self, target_epsilon_consume, datablock_epsilon_capacity, epsilon_star):
        if self.comparison_cost_epsilon > 0.0:
            if epsilon_star + self.Lap(4.0/self.comparison_cost_epsilon) >= self.gitta * target_epsilon_consume + self.Lap(2.0/self.comparison_cost_epsilon):
                is_select = True
                compare_epsilon = self.comparison_cost_epsilon
            else:
                is_select = False
                compare_epsilon = 0.0
        else:
            if epsilon_star >= self.gitta * target_epsilon_consume:
                is_select = True
                compare_epsilon = 0.0
            else:
                is_select = False
                compare_epsilon = 0.0
        real_sched_epsilon = min(epsilon_star, datablock_epsilon_capacity - compare_epsilon, target_epsilon_consume)
        return is_select, real_sched_epsilon, compare_epsilon
        
    def get_allocation(self, state):
        job_id_2_train_dataset_name = state["job_id_2_train_dataset_name"]
        assert len(job_id_2_train_dataset_name) == 1
        set_job_id = set(job_id_2_train_dataset_name.keys())
        set_dataset_name = set(job_id_2_train_dataset_name.values())
        assert len(set_dataset_name) == 1 # 必须保证所有的任务都是针对同一个数据集的
        job_id = list(set_job_id)[0]
        train_dataset_name = list(set_dataset_name)[0]
        
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
        selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        calcu_compare_epsilon = 0.0
        
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取随机一个数据集
            datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]
            significance_plus_weight = job_priority_weight * sub_train_datasetidentifier_2_significance[datasetidentifier]
            T = self.U / (1 + np.log(self.U / self.L))
            if self.comparison_cost_epsilon > 0.0:
                epsilon_star = significance_plus_weight / T - self.comparison_cost_epsilon
            else:
                epsilon_star = significance_plus_weight / T
            
            is_select, real_sched_epsilon, compare_epsilon = self.filter_by_threshold(target_epsilon_require, datablock_epsilon_capacity, epsilon_star)
            if is_select:
                count += 1
                selected_datablock_identifiers.append(datasetidentifier)
                selected_real_sched_epsilon_map[(job_id, datasetidentifier)] = real_sched_epsilon
                calcu_compare_epsilon += compare_epsilon
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
        
        job_2_selected_datablock_identifiers = [
            (job_id, identifier) for identifier in selected_datablock_identifiers
        ]
        
        return job_2_selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon

