from policies.HISBase import HISBasePolicy
import copy
import random
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import cvxpy as cp
import json

class HISwithCPolicy(HISBasePolicy):
    def __init__(self, beta, job_sequence_all_num, seed, logger):
        super().__init__(beta, job_sequence_all_num, seed, logger)
        self._name = 'HISwithCPolicy'
        self.beta = beta
        # self.gamma = gamma
        # self.delta = delta
        # self.only_small = only_small
        self.logger = logger
        self.waiting_queue_capacity = 1
        self.only_one = True
        self.need_history = True
        self.initialize_seeds(seed)

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

    def get_LP_result(self, sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, job_privacy_budget_consume_list, solver=cp.ECOS):
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        job_target_datablock_selected_num_list = np.array(job_target_datablock_selected_num_list)
        job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
        datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

        matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
        objective = cp.Maximize(
            cp.sum(cp.sum(cp.multiply(sign_matrix, matrix_X), axis=1) / job_target_datablock_selected_num_list)
        )

        constraints = [
            matrix_X >= 0,
            matrix_X <= 1,
            cp.sum(matrix_X, axis=1) <= job_target_datablock_selected_num_list,
            (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
        ]

        # self.logger.debug("check sign_matrix: {}".format(sign_matrix))
        # self.logger.debug("check job_privacy_budget_consume_list: {}".format(job_privacy_budget_consume_list))
        # self.logger.debug("check job_target_datablock_selected_num_list: {}".format(job_target_datablock_selected_num_list))
        # self.logger.debug("check datablock_privacy_budget_capacity_list: {}".format(datablock_privacy_budget_capacity_list))
        # self.logger.debug("check sum of datablock_privacy_budget_capacity_list: {}".format(np.sum(datablock_privacy_budget_capacity_list)))
        # self.logger.debug("check sum of job_privacy_budget_consume_list: {}".format(np.sum(job_privacy_budget_consume_list * job_target_datablock_selected_num_list)))

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver)
        # self.logger.debug(matrix_X.value)
        if cvxprob.status != "optimal":
            self.logger.info('WARNING: Allocation returned by policy not optimal!')
        return matrix_X.value

    def get_allocation_for_small(self, job_id, history_job_priority_weights, 
                                history_job_budget_consumes, 
                                history_job_signficances, 
                                history_job_target_datablock_selected_nums,
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                target_epsilon_require, 
                                target_datablock_select_num, 
                                job_priority_weight):
        
        selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        calcu_compare_epsilon = 0.0
        
        current_all_job_priority_weights = copy.deepcopy(history_job_priority_weights)
        current_all_job_priority_weights.append(job_priority_weight)
        current_all_job_budget_consumes = copy.deepcopy(history_job_budget_consumes)
        current_all_job_budget_consumes.append(target_epsilon_require)
        current_all_job_signficances = copy.deepcopy(history_job_signficances)
        current_all_job_signficances.append(sub_train_datasetidentifier_2_significance)
        current_all_job_target_datablock_selected_nums = copy.deepcopy(history_job_target_datablock_selected_nums)
        current_all_job_target_datablock_selected_nums.append(target_datablock_select_num)

        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            current_all_job_priority_weights,
            current_all_job_signficances,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        for temp_index in temp_index_2_datablock_identifier:
            datablock_privacy_budget_capacity_list[temp_index] = sub_train_datasetidentifier_2_epsilon_capcity[temp_index_2_datablock_identifier[temp_index]]
        assign_result_matrix = self.get_LP_result(
            sign_matrix, 
            datablock_privacy_budget_capacity_list, 
            current_all_job_target_datablock_selected_nums, 
            current_all_job_budget_consumes
        )
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        current_job_probability = assign_result_matrix[-1] # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
        choose_indexes = []
        waiting_select_indexes = np.array(range(datablock_num + 1))
        current_job_probability = list(current_job_probability)
        current_job_probability.append(target_datablock_select_num - sum(current_job_probability))
        null_index = len(current_job_probability) - 1
        for index, proba in enumerate(current_job_probability):
            if proba <= 0.0:
                current_job_probability[index] = 0.0
            if index < null_index:
                datablock_identifier = temp_index_2_datablock_identifier[index]
                if target_epsilon_require > sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                    current_job_probability[index] = 0.0

        sum_current_job_probability = sum(current_job_probability)
        if sum_current_job_probability > 1e-17:
            current_job_probability = np.divide(current_job_probability, sum_current_job_probability)
        else:
            current_job_probability[null_index] = 1.0
            temp_sum_current_job_probability = sum(current_job_probability)
            current_job_probability = np.divide(current_job_probability, temp_sum_current_job_probability)
        result_select_num = target_datablock_select_num
        if len(waiting_select_indexes) < target_datablock_select_num:
            result_select_num = len(waiting_select_indexes)
        probability_enable_num = sum(p > 0.0 for p in current_job_probability)
        if probability_enable_num < result_select_num:
            result_select_num = probability_enable_num
        # self.logger.debug("check result_select_num: ", result_select_num)
        # self.logger.debug("check waiting_select_indexes: ", waiting_select_indexes)
        # self.logger.debug("check current_job_probability: {}".format(current_job_probability))
        temp_result = list(np.random.choice(a=waiting_select_indexes, size=result_select_num, replace=False, p=current_job_probability))
        if null_index in temp_result:
            choose_indexes = copy.deepcopy(temp_result)
            choose_indexes.remove(null_index)
        else:
            choose_indexes = copy.deepcopy(temp_result)
        for choose_index in choose_indexes:
            datablock_identifier = temp_index_2_datablock_identifier[choose_index]
            if target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                selected_datablock_identifiers.append(datablock_identifier)
                selected_real_sched_epsilon_map[(job_id, datablock_identifier)] = target_epsilon_require
        return selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon

    def get_allocation(self, state):
        job_id, train_dataset_name = self.get_allocation_judge_one_job(state)

        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
        target_epsilon_require = state["job_id_2_target_epsilon_require"][job_id]
        target_datablock_select_num = state["job_id_2_target_datablock_selected_num"][job_id]
        job_priority_weight = state["job_id_2_job_priority_weight"][job_id]
        sub_train_datasetidentifier_2_significance = state["job_id_2_significance"][job_id]
        job_arrival_index = state["job_id_2_arrival_index"][job_id]
        
        all_job_sequence_num = self.job_sequence_all_num
        offline_history_job_priority_weights = self.offline_history_job_priority_weights
        offline_history_job_budget_consumes = self.offline_history_job_budget_consumes
        offline_history_job_signficance = self.offline_history_job_significance
        offline_history_job_target_datablock_selected_num = self.offline_history_job_target_selected_num

        online_history_job_priority_weights = self.online_history_job_priority_weights
        online_history_job_budget_consumes = self.online_history_job_budget_consumes
        online_history_job_signficance = self.online_history_job_significance
        online_history_job_target_datablock_selected_num = self.online_history_job_target_selected_num
        # assert target_datablock_select_num == 1
        
        if len(offline_history_job_priority_weights) + len(online_history_job_priority_weights) < all_job_sequence_num:
            sample_history_job_priority_weights = offline_history_job_priority_weights + online_history_job_priority_weights
            sample_history_job_budget_consumes = offline_history_job_budget_consumes + online_history_job_budget_consumes
            sample_history_job_signficances = offline_history_job_signficance + online_history_job_signficance
            sample_history_job_target_datablock_selected_nums = offline_history_job_target_datablock_selected_num + online_history_job_target_datablock_selected_num
        else:
            select_num_from_offline_history = max(self.job_sequence_all_num - len(online_history_job_priority_weights) - 1, 0)
            offline_sample_indexes = np.random.choice(range(len(offline_history_job_priority_weights)), select_num_from_offline_history, replace=False)
            
            if len(online_history_job_priority_weights) > self.job_sequence_all_num - 1:
                online_sample_indexes = np.random.choice(range(len(online_history_job_priority_weights)), self.job_sequence_all_num - 1, replace=False)
            else:
                online_sample_indexes = range(len(online_history_job_priority_weights))
            sample_history_job_priority_weights = [online_history_job_priority_weights[i] for i in online_sample_indexes] + [offline_history_job_priority_weights[i] for i in offline_sample_indexes]
            sample_history_job_budget_consumes = [online_history_job_budget_consumes[i] for i in online_sample_indexes] + [offline_history_job_budget_consumes[i] for i in offline_sample_indexes]
            sample_history_job_signficances = [online_history_job_signficance[i] for i in online_sample_indexes] + [offline_history_job_signficance[i] for i in offline_sample_indexes]
            sample_history_job_target_datablock_selected_nums = [online_history_job_target_datablock_selected_num[i] for i in online_sample_indexes] + [offline_history_job_target_datablock_selected_num[i] for i in offline_sample_indexes]

        if job_arrival_index < self.beta * all_job_sequence_num:
            self.logger.info("stop due to sample caused by job_arrival_index: {}; self.beta: {}; all_job_sequence_num: {}".format(
                job_arrival_index, self.beta, all_job_sequence_num
            ))
            selected_datablock_identifiers = []
            selected_real_sched_epsilon_map = {}
            calcu_compare_epsilon = 0.0
        else:
            selected_datablock_identifiers, selected_real_sched_epsilon_map, \
                calcu_compare_epsilon = self.get_allocation_for_small(job_id, sample_history_job_priority_weights, 
                                sample_history_job_budget_consumes, 
                                sample_history_job_signficances, 
                                sample_history_job_target_datablock_selected_nums,
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                target_epsilon_require, target_datablock_select_num, job_priority_weight)
   
        job_2_selected_datablock_identifiers = [
            (job_id, identifier) for identifier in selected_datablock_identifiers
        ]
        waiting_job_ids = []
        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name , job_2_selected_datablock_identifiers))
        return job_2_selected_datablock_identifiers, waiting_job_ids, selected_real_sched_epsilon_map, calcu_compare_epsilon