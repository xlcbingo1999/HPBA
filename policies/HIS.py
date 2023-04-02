from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import cvxpy as cp
import json

class HISPolicy(Policy):
    def __init__(self, beta, logger):
        super().__init__()
        self._name = 'HISPolicy'
        self.beta = beta
        # self.gamma = gamma
        # self.delta = delta
        # self.only_small = only_small
        self.logger = logger
        self.waiting_queue_capacity = 1
        self.only_one = True

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))
        # self.logger.info("policy args: gamma: {}".format(self.gamma))
        # self.logger.info("policy args: delta: {}".format(self.delta))
        # self.logger.info("policy args: only_small: {}".format(self.only_small))

    def get_LP_result(self, sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, job_privacy_budget_consume_list, solver=cp.ECOS):
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
        datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

        matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
        objective = cp.Maximize(
            cp.sum(cp.multiply(sign_matrix, matrix_X))
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
        self.logger.debug("check sum of job_privacy_budget_consume_list: {}".format(np.sum(job_privacy_budget_consume_list * job_target_datablock_selected_num_list)))

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver)
        # self.logger.debug(matrix_X.value)
        if cvxprob.status != "optimal":
            self.logger.info('WARNING: Allocation returned by policy not optimal!')
        return matrix_X.value

    def get_sign_matrix(self, current_all_job_priority_weights, current_all_job_significances,
                        sub_train_datasetidentifier_2_epsilon_capcity):
        temp_index_2_datablock_identifier = {}
        sign_matrix = []
        for job_index, job_priority_weight in enumerate(current_all_job_priority_weights):
            temp = []
            for datablock_index, datablock_identifier in enumerate(sub_train_datasetidentifier_2_epsilon_capcity):
                temp_index_2_datablock_identifier[datablock_index] = datablock_identifier
                temp.append(current_all_job_significances[job_index][datablock_identifier] * job_priority_weight)
            sign_matrix.append(temp)
        sign_matrix = np.array(sign_matrix)
        return sign_matrix, temp_index_2_datablock_identifier

    '''
    def get_allocation_for_large(self, history_job_priority_weights, sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity,
                                index, target_epsilon_require, target_datablock_select_num, job_priority_weight):
        # assert target_datablock_select_num == 1
        
        selected_datablock_identifiers = []
        calcu_compare_epsilon = 0.0
        
        current_all_job_significances = copy.deepcopy(history_job_priority_weights) 
        current_all_job_significances.append(job_priority_weight)
        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            True, current_all_job_significances, sub_train_datasetidentifier_2_significance,
            sub_train_datasetidentifier_2_epsilon_capcity, target_epsilon_require, job_priority_weight
        )
        row_inds, col_inds = linear_sum_assignment(sign_matrix)
        if index in row_inds:
            target_datablock_index = col_inds[index]
            target_datablock_identifier = temp_index_2_datablock_identifier[target_datablock_index]
            if target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[target_datablock_identifier]:
                selected_datablock_identifiers.append(target_datablock_identifier)
        
        return selected_datablock_identifiers, calcu_compare_epsilon
    '''

    def get_allocation_for_small(self, history_job_priority_weights, 
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
        job_target_datablock_selected_num_list = np.array(current_all_job_target_datablock_selected_nums)
        for temp_index in temp_index_2_datablock_identifier:
            datablock_privacy_budget_capacity_list[temp_index] = sub_train_datasetidentifier_2_epsilon_capcity[temp_index_2_datablock_identifier[temp_index]]
        assign_result_matrix = self.get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, current_all_job_budget_consumes)
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

        current_job_probability = current_job_probability / sum(current_job_probability)
        result_select_num = target_datablock_select_num
        if len(waiting_select_indexes) < target_datablock_select_num:
            result_select_num = len(waiting_select_indexes)
        probability_enable_num = sum(p > 0.0 for p in current_job_probability)
        if probability_enable_num < result_select_num:
            result_select_num = probability_enable_num
        # self.logger.debug("check result_select_num: ", result_select_num)
        # self.logger.debug("check waiting_select_indexes: ", waiting_select_indexes)
        self.logger.debug("check current_job_probability: {}".format(current_job_probability))
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
        return selected_datablock_identifiers, calcu_compare_epsilon

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
        job_arrival_index = state["job_id_2_arrival_index"][job_id]
        
        all_job_sequence_num = state["all_job_sequence_num"]
        history_job_priority_weights = state["history_job_priority_weights"]
        history_job_budget_consumes = state["history_job_budget_consumes"]
        history_job_signficance = state["history_job_significance"]
        history_job_target_datablock_selected_num = state["history_job_target_datablock_selected_num"]

        # assert target_datablock_select_num == 1
        
        if len(history_job_priority_weights) < all_job_sequence_num:
            sample_history_job_priority_weights = history_job_priority_weights
            sample_history_job_budget_consumes = history_job_budget_consumes
            sample_history_job_signficances = history_job_signficance
            sample_history_job_target_datablock_selected_nums = history_job_target_datablock_selected_num
        else:
            sample_indexes = random.sample(range(len(history_job_priority_weights)), all_job_sequence_num-1)
            sample_history_job_priority_weights = [history_job_priority_weights[i] for i in sample_indexes]
            sample_history_job_budget_consumes = [history_job_budget_consumes[i] for i in sample_indexes]
            sample_history_job_signficances = [history_job_signficance[i] for i in sample_indexes]
            sample_history_job_target_datablock_selected_nums = [history_job_target_datablock_selected_num[i] for i in sample_indexes]

        if job_arrival_index < self.beta * all_job_sequence_num:
            self.logger.info("stop due to sample caused by job_arrival_index: {}; self.beta: {}; all_job_sequence_num: {}".format(
                job_arrival_index, self.beta, all_job_sequence_num
            ))
            selected_datablock_identifiers = []
            calcu_compare_epsilon = 0.0
        else:
            selected_datablock_identifiers, \
                calcu_compare_epsilon = self.get_allocation_for_small(sample_history_job_priority_weights, 
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
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon