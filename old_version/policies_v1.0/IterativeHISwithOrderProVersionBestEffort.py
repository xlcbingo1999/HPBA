from policies.HISBase import HISBasePolicy
import copy
import random
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import cvxpy as cp
import json
from queue import PriorityQueue
import time

class QueueItem(object):
    def __init__(self, job_id, datablock_identifier, significance):
        self.job_id = job_id
        self.datablock_identifier = datablock_identifier 
        self.significance = significance

class IterativeHISwithOrderProVersionBestEffortPolicy(HISBasePolicy):
    def __init__(self, beta, job_sequence_all_num, batch_size_for_one_epoch, seed, logger):
        super().__init__(beta, job_sequence_all_num, seed, logger)
        self._name = 'IterativeHISwithOrderProVersionBestEffortPolicy'
        self.beta = beta
        self.logger = logger
        self.waiting_queue_capacity = 1
        self.only_one = True
        self.need_history = True
        self.initialize_seeds(seed)

        self.job_sequence_all_num = job_sequence_all_num
        self.batch_size_for_one_epoch = batch_size_for_one_epoch
        self.all_epoch_num = np.ceil(self.job_sequence_all_num / self.batch_size_for_one_epoch)
        self.logger.info("check job_sequence_all_num: {}".format(self.job_sequence_all_num))

        self.current_epoch_index = 0
        self.current_batch_size_for_one_epoch = 0
        self.datablock_identifier_2_all_epoch_num = {}
        self.datablock_identifier_2_epsilon_G = {}
        self.datablock_identifier_2_remain_epsilon = {}

        self.best_effort_jobs_queue = PriorityQueue()

        self.offline_history_job_priority_weights = []
        self.offline_history_job_budget_consumes = []
        self.offline_history_job_target_selected_num = []
        self.offline_history_job_train_dataset_name = []
        self.offline_history_job_test_dataset_name = []
        self.offline_history_job_sub_test_key_id = []
        self.offline_history_job_type_id = []
        self.offline_history_job_significance = []
        self.offline_history_job_arrival_time = []

        self.online_history_job_priority_weights = []
        self.online_history_job_budget_consumes = []
        self.online_history_job_target_selected_num = []
        self.online_history_job_train_dataset_name = []
        self.online_history_job_test_dataset_name = []
        self.online_history_job_sub_test_key_id = []
        self.online_history_job_type_id = []
        self.online_history_job_significance = []
        self.online_history_job_arrival_time = []

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))
        self.logger.info("policy args: all_epoch_num: {}".format(self.all_epoch_num))
        self.logger.info("policy args: batch_size_for_one_epoch: {}".format(self.batch_size_for_one_epoch))

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

    def get_LP_result(self, sign_matrix, 
                      datablock_privacy_budget_capacity_list, 
                      datablock_arrival_time_list,
                      job_target_datablock_selected_num_list, 
                      job_privacy_budget_consume_list, 
                      job_arrival_time_list,
                      solver=cp.ECOS):
        begin_time = time.time()
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        job_target_datablock_selected_num_list = np.array(job_target_datablock_selected_num_list)
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
        for job_index, job_arrival_time in enumerate(job_arrival_time_list):
            for datablock_index, datablock_arrival_time in enumerate(datablock_arrival_time_list):
                if job_arrival_time < datablock_arrival_time:
                    constraints.append(matrix_X[job_index, datablock_index] == 0)
        
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver)
        # self.logger.debug(matrix_X.value)
        if cvxprob.status != "optimal":
            self.logger.info('WARNING: Allocation returned by policy not optimal!')
        self.logger.debug("LP solver time: {} s".format(time.time() - begin_time))
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

    def get_allocation_for_small(self, job_id, history_job_priority_weights, 
                                history_job_budget_consumes, 
                                history_job_signficances, 
                                history_job_target_datablock_selected_nums,
                                history_job_arrival_times,
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                sub_train_datasetidentifier_2_arrival_time,
                                target_epsilon_require, 
                                target_datablock_select_num, 
                                target_arrival_time,
                                job_priority_weight,
                                job_arrival_index, 
                                all_job_sequence_num):
        
        selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        calcu_compare_epsilon = 0.0
        
        current_all_job_priority_weights = history_job_priority_weights
        current_all_job_priority_weights.append(job_priority_weight)
        current_all_job_budget_consumes = history_job_budget_consumes
        current_all_job_budget_consumes.append(target_epsilon_require)
        current_all_job_signficances = history_job_signficances
        current_all_job_signficances.append(sub_train_datasetidentifier_2_significance)
        current_all_job_target_datablock_selected_nums = history_job_target_datablock_selected_nums
        current_all_job_target_datablock_selected_nums.append(target_datablock_select_num)
        current_all_job_arrival_times = history_job_arrival_times
        current_all_job_arrival_times.append(target_arrival_time)
        

        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            current_all_job_priority_weights,
            current_all_job_signficances,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        datablock_privacy_budget_remain_list = np.zeros(shape=sign_matrix.shape[1])
        datablock_arrival_time_list = np.zeros(shape=sign_matrix.shape[1])
        job_arrival_time_list = np.array(current_all_job_arrival_times)
        
        for temp_index, datablock_identifier in temp_index_2_datablock_identifier.items():
            datablock_privacy_budget_capacity_list[temp_index] = self.datablock_identifier_2_remain_epsilon[datablock_identifier] + \
                sub_train_datasetidentifier_2_epsilon_capcity[datablock_identifier] / self.datablock_identifier_2_all_epoch_num[datablock_identifier]
            datablock_privacy_budget_remain_list[temp_index] = sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]
            datablock_arrival_time_list[temp_index] = sub_train_datasetidentifier_2_arrival_time[datablock_identifier]
        
        assign_result_matrix = self.get_LP_result(sign_matrix, 
                                                datablock_privacy_budget_capacity_list,
                                                datablock_arrival_time_list, 
                                                current_all_job_target_datablock_selected_nums,
                                                current_all_job_budget_consumes,
                                                current_all_job_arrival_times)
        current_job_probability = assign_result_matrix[-1] # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
        choose_indexes = []
        
        current_job_probability_list = list(current_job_probability)
        current_job_probability_sorted_indexes = sorted(range(len(current_job_probability_list)), key=lambda k: (datablock_privacy_budget_remain_list[k], current_job_probability_list[k]), reverse=True)
        
        for sorted_index in current_job_probability_sorted_indexes:
            if sorted_index in choose_indexes:
                continue
            prob_true = min(1.0, max(0.0, current_job_probability_list[sorted_index]))
            if datablock_privacy_budget_remain_list[sorted_index] < target_epsilon_require:
                prob_true = 0.0
            prob_false = 1.0 - prob_true
            prob_vec = [prob_false, prob_true]
            choice_result = np.random.choice(a=range(2), size=1, replace=False, p=prob_vec)
            if choice_result == 1:
                choose_indexes.append(sorted_index)

            self.logger.debug(f"(job_id[{job_id}], datablock_identifier[{temp_index_2_datablock_identifier[sorted_index]}]) => remain: {datablock_privacy_budget_remain_list[sorted_index]}; pro: {current_job_probability_list[sorted_index]}; choice_result: {choice_result}")
        
        self.logger.debug(f"job_id[{job_id}] step[pro]: choose_indexes: {choose_indexes}")

        for choose_index in choose_indexes:
            datablock_identifier = temp_index_2_datablock_identifier[choose_index]
            if len(selected_datablock_identifiers) < target_datablock_select_num and target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                selected_datablock_identifiers.append(datablock_identifier)
                selected_real_sched_epsilon_map[(job_id, datablock_identifier)] = target_epsilon_require
        return selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon

    def get_allocation(self, state, all_or_nothing_flag, enable_waiting_flag):
        job_id, train_dataset_name = self.get_allocation_judge_one_job(state)

        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
        sub_train_datasetidentifier_2_arrival_time = state["current_sub_train_datasetidentifier_2_arrival_time"][train_dataset_name]
        target_epsilon_require = state["job_id_2_target_epsilon_require"][job_id]
        target_datablock_select_num = state["job_id_2_target_datablock_selected_num"][job_id]
        target_arrival_time = state["job_id_2_arrival_time"][job_id]
        job_priority_weight = state["job_id_2_job_priority_weight"][job_id]
        sub_train_datasetidentifier_2_significance = state["job_id_2_significance"][job_id]
        job_arrival_index = state["job_id_2_arrival_index"][job_id]
        
        all_job_sequence_num = self.job_sequence_all_num
        offline_history_job_priority_weights = self.offline_history_job_priority_weights
        offline_history_job_budget_consumes = self.offline_history_job_budget_consumes
        offline_history_job_signficance = self.offline_history_job_significance
        offline_history_job_target_datablock_selected_num = self.offline_history_job_target_selected_num
        offline_history_job_arrival_time = self.offline_history_job_arrival_time

        online_history_job_priority_weights = self.online_history_job_priority_weights
        online_history_job_budget_consumes = self.online_history_job_budget_consumes
        online_history_job_signficance = self.online_history_job_significance
        online_history_job_target_datablock_selected_num = self.online_history_job_target_selected_num
        online_history_job_arrival_time = self.online_history_job_arrival_time

        # 处理一下新来的数据块
        for sub_train_dataset_identifier in sub_train_datasetidentifier_2_epsilon_capcity:
            if sub_train_dataset_identifier not in self.datablock_identifier_2_epsilon_G:
                self.datablock_identifier_2_epsilon_G[sub_train_dataset_identifier] = sub_train_datasetidentifier_2_epsilon_capcity[sub_train_dataset_identifier]
                self.datablock_identifier_2_all_epoch_num[sub_train_dataset_identifier] = (self.all_epoch_num - self.current_epoch_index)
                self.datablock_identifier_2_remain_epsilon[sub_train_dataset_identifier] = self.datablock_identifier_2_epsilon_G[sub_train_dataset_identifier] / self.datablock_identifier_2_all_epoch_num[sub_train_dataset_identifier]
        
        if self.current_batch_size_for_one_epoch >= self.batch_size_for_one_epoch: # 每个epoch的最后一个batch进行best-effort和remain epsilon budget的计算
            self.current_epoch_index += 1
            self.current_batch_size_for_one_epoch = 0

            # TODO(xlc): 计算remain epsilon budget!
            for sub_train_dataset_identifier in sub_train_datasetidentifier_2_epsilon_capcity:
                self.datablock_identifier_2_remain_epsilon[sub_train_dataset_identifier] += (self.datablock_identifier_2_epsilon_G[sub_train_dataset_identifier] / self.datablock_identifier_2_all_epoch_num[sub_train_dataset_identifier])
            
            self.logger.info("update datablock_identifier_2_remain_epsilon: {}".format(self.datablock_identifier_2_remain_epsilon))

        self.current_batch_size_for_one_epoch += 1
        if len(offline_history_job_priority_weights) + len(online_history_job_priority_weights) < self.batch_size_for_one_epoch:
            sample_history_job_priority_weights = offline_history_job_priority_weights + online_history_job_priority_weights
            sample_history_job_budget_consumes = offline_history_job_budget_consumes + online_history_job_budget_consumes
            sample_history_job_signficances = offline_history_job_signficance + online_history_job_signficance
            sample_history_job_target_datablock_selected_nums = offline_history_job_target_datablock_selected_num + online_history_job_target_datablock_selected_num
            sample_history_job_arrival_times = offline_history_job_arrival_time + online_history_job_arrival_time
        else:
            select_num_from_offline_history = max(self.batch_size_for_one_epoch - len(online_history_job_priority_weights) - 1, 0)
            offline_sample_indexes = np.random.choice(range(len(offline_history_job_priority_weights)), select_num_from_offline_history, replace=False)
            
            if len(online_history_job_priority_weights) > self.batch_size_for_one_epoch - 1:
                online_sample_indexes = np.random.choice(range(len(online_history_job_priority_weights)), self.batch_size_for_one_epoch - 1, replace=False)
            else:
                online_sample_indexes = range(len(online_history_job_priority_weights))
            sample_history_job_priority_weights = [online_history_job_priority_weights[i] for i in online_sample_indexes] + [offline_history_job_priority_weights[i] for i in offline_sample_indexes]
            sample_history_job_budget_consumes = [online_history_job_budget_consumes[i] for i in online_sample_indexes] + [offline_history_job_budget_consumes[i] for i in offline_sample_indexes]
            sample_history_job_signficances = [online_history_job_signficance[i] for i in online_sample_indexes] + [offline_history_job_signficance[i] for i in offline_sample_indexes]
            sample_history_job_target_datablock_selected_nums = [online_history_job_target_datablock_selected_num[i] for i in online_sample_indexes] + [offline_history_job_target_datablock_selected_num[i] for i in offline_sample_indexes]
            sample_history_job_arrival_times = [online_history_job_arrival_time[i] for i in online_sample_indexes] + [offline_history_job_arrival_time[i] for i in offline_sample_indexes]

        if (not enable_waiting_flag) and job_arrival_index % self.batch_size_for_one_epoch < self.beta * self.batch_size_for_one_epoch:
            self.logger.info("stop due to sample caused by job_arrival_index: {}; self.beta: {}; all_job_sequence_num: {}".format(
                job_arrival_index, self.beta, all_job_sequence_num
            ))
            selected_datablock_identifiers = []
            selected_real_sched_epsilon_map = {}
            calcu_compare_epsilon = 0.0
        else:
            selected_datablock_identifiers, selected_real_sched_epsilon_map, \
                calcu_compare_epsilon = self.get_allocation_for_small(job_id, 
                                sample_history_job_priority_weights, 
                                sample_history_job_budget_consumes, 
                                sample_history_job_signficances, 
                                sample_history_job_target_datablock_selected_nums,
                                sample_history_job_arrival_times,
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                sub_train_datasetidentifier_2_arrival_time,
                                target_epsilon_require, 
                                target_datablock_select_num, 
                                target_arrival_time,
                                job_priority_weight,
                                job_arrival_index, 
                                all_job_sequence_num)
   
        job_2_selected_datablock_identifiers = [
            (job_id, identifier) for identifier in selected_datablock_identifiers
        ]
        waiting_job_ids = []
        need_waiting_job_sched 
        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name , job_2_selected_datablock_identifiers))
        return job_2_selected_datablock_identifiers, waiting_job_ids, selected_real_sched_epsilon_map, calcu_compare_epsilon
    
    def push_success_allocation(self, success_datasetidentifier_2_consume_epsilon):
        if len(success_datasetidentifier_2_consume_epsilon.keys()) <= 0:
            return 
        assert len(success_datasetidentifier_2_consume_epsilon.keys()) == 1
        dataset_name = list(success_datasetidentifier_2_consume_epsilon.keys())[0]
        datablock_identifier_2_consume_epsilon = success_datasetidentifier_2_consume_epsilon[dataset_name]
        
        for datablock_identifier, consume_epsilon in datablock_identifier_2_consume_epsilon.items():
            if self.datablock_identifier_2_remain_epsilon[datablock_identifier] >= consume_epsilon:
                self.datablock_identifier_2_remain_epsilon[datablock_identifier] -= consume_epsilon
            else:
                self.datablock_identifier_2_remain_epsilon[datablock_identifier] = 0.0
            if self.datablock_identifier_2_remain_epsilon[datablock_identifier] < 0:
                self.logger.warning("datablock_identifier_2_remain_epsilon[{}] == {}".format(
                    datablock_identifier, self.datablock_identifier_2_remain_epsilon[datablock_identifier]
                ))