from policies.HISBase import HISBasePolicy
import copy
import random
import numpy as np
import math
import cvxpy as cp
import json
import sys

class HISwithOrderProVersionPolicy(HISBasePolicy):
    def __init__(self, beta, pipeline_sequence_all_num, job_request_all_num, datablocks_privacy_budget_all,
                infinity_flag, adaptive_n_flag,
                greedy_flag, greedy_threshold,
                seed, logger):
        super().__init__(beta, pipeline_sequence_all_num, job_request_all_num, 
                        infinity_flag, 
                        greedy_flag, greedy_threshold,
                        seed, logger)
        self._name = 'HISwithOrderProVersionPolicy'
        self.beta = beta

        self.adaptive_n_flag = adaptive_n_flag
        self.adaptive_offline_h_num = 0
        self.datablocks_privacy_budget_all = datablocks_privacy_budget_all

        self.logger = logger
        self.waiting_queue_capacity = 1
        self.only_one = True
        self.need_history = True
        self.initialize_seeds(seed)

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))
        self.logger.info("policy args: adaptive_n_flag: {}".format(self.adaptive_n_flag))
        self.logger.info("policy args: job_request_all_num: {}".format(self.job_request_all_num))
        # self.logger.info("policy args: delta: {}".format(self.delta))
        # self.logger.info("policy args: only_small: {}".format(self.only_small))

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

    def get_LP_result(self, sign_matrix, 
                      datablock_privacy_budget_capacity_list,
                      datablock_privacy_budget_remain_list, 
                      datablock_arrival_time_list,
                      job_target_datablock_selected_num_list, 
                      job_privacy_budget_consume_list, 
                      job_arrival_time_list,
                      all_or_nothing_flag, 
                      enable_waiting_flag,
                      solver=cp.ECOS):
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        valid_sched_datablock_indices = np.where(datablock_privacy_budget_remain_list >= job_privacy_budget_consume_list[-1])[0]
        if len(valid_sched_datablock_indices) <= 0:
            self.logger.debug(f"len(valid_sched_datablock_indices) <= 0")
            return np.zeros((job_num, datablock_num))  
            
        # 检查本身就无法调度上岸的方案
        invalid_sched_datablock_indices = np.where(datablock_privacy_budget_remain_list < job_privacy_budget_consume_list[-1])[0]
        
        job_target_datablock_selected_num_list = np.array(job_target_datablock_selected_num_list)
        job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
        datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

        self.logger.debug(f"HIS check solving shape: ({job_num}, {datablock_num})")
        matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
        objective = cp.Maximize(
            cp.sum(cp.multiply(sign_matrix, matrix_X))
        )

        constraints = [
            matrix_X >= 0,
            matrix_X <= 1,
            (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
        ]

        for datablock_index in invalid_sched_datablock_indices:
            constraints.append(matrix_X[-1, datablock_index] == 0)
        if all_or_nothing_flag:
            # TODO(xlc): 直接用这个放松版本应该也可以过来, 毕竟离线最优IP解project后的部分也是这个放松后问题的直接解法, 但是可能Offline不好求!
            # vector_Y = cp.Variable((job_num, ), boolean=True)
            # constraints.append(cp.sum(matrix_X, axis=1) == cp.multiply(vector_Y, job_target_datablock_selected_num_list))
            constraints.append(cp.sum(matrix_X, axis=1) <= job_target_datablock_selected_num_list)
        else:
            constraints.append(cp.sum(matrix_X, axis=1) <= job_target_datablock_selected_num_list)
        

        # TODO(xlc): 对于HIS和IterativeHIS是不需要加入额外的约束的
        # if not enable_waiting_flag:
        #     add_time_constraint_num = 0
        #     for job_index, job_arrival_time in enumerate(job_arrival_time_list):
        #         for datablock_index, datablock_arrival_time in enumerate(datablock_arrival_time_list):
        #             if job_arrival_time > datablock_arrival_time:
        #                 constraints.append(matrix_X[job_index, datablock_index] == 0)
        #                 add_time_constraint_num += 1
        #     self.logger.debug(f"add_time_constraint_num: {add_time_constraint_num}")

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver)
        # self.logger.debug(matrix_X.value)
        if cvxprob.status != "optimal":
            self.logger.info('WARNING: Allocation returned by policy not optimal!')
        return matrix_X.value

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

    def get_allocation_for_small(self, job_id, 
                                history_job_ids,
                                history_job_priority_weights, 
                                history_job_budget_consumes, 
                                history_job_signficances, 
                                history_job_target_datablock_selected_nums,
                                history_job_arrival_times,
                                history_job_test_dataset_names,
                                history_job_sub_test_key_ids,
                                history_job_train_dataset_names,
                                history_job_model_names,
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                sub_train_datasetidentifier_2_arrival_time,
                                target_epsilon_require, 
                                target_datablock_select_num, 
                                target_arrival_time,
                                job_priority_weight,
                                job_test_dataset_name,
                                job_sub_test_key_id,
                                job_train_dataset_name,
                                job_model_name,
                                all_or_nothing_flag, 
                                enable_waiting_flag):
        
        selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        calcu_compare_epsilon = 0.0
        
        current_all_job_ids = history_job_ids
        current_all_job_ids.append(job_id)
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

        current_all_job_test_dataset_names = history_job_test_dataset_names
        current_all_job_test_dataset_names.append(job_test_dataset_name)
        current_all_job_sub_test_key_ids = history_job_sub_test_key_ids
        current_all_job_sub_test_key_ids.append(job_sub_test_key_id)
        current_all_job_train_dataset_names = history_job_train_dataset_names
        current_all_job_train_dataset_names.append(job_train_dataset_name)
        current_all_job_model_names = history_job_model_names
        current_all_job_model_names.append(job_model_name)

        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            current_all_job_ids,
            current_all_job_priority_weights,
            current_all_job_signficances,
            current_all_job_test_dataset_names,
            current_all_job_sub_test_key_ids,
            current_all_job_train_dataset_names,
            current_all_job_model_names,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        datablock_privacy_budget_remain_list = np.zeros(shape=sign_matrix.shape[1])
        datablock_arrival_time_list = np.zeros(shape=sign_matrix.shape[1])

        for temp_index, datablock_identifier in temp_index_2_datablock_identifier.items():
            datablock_privacy_budget_capacity_list[temp_index] = sub_train_datasetidentifier_2_epsilon_capcity[datablock_identifier]
            datablock_privacy_budget_remain_list[temp_index] = sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]
            datablock_arrival_time_list[temp_index] = sub_train_datasetidentifier_2_arrival_time[datablock_identifier]

        assign_result_matrix = self.get_LP_result(sign_matrix, 
                                                datablock_privacy_budget_capacity_list,
                                                datablock_privacy_budget_remain_list,
                                                datablock_arrival_time_list, 
                                                current_all_job_target_datablock_selected_nums,
                                                current_all_job_budget_consumes,
                                                current_all_job_arrival_times,
                                                all_or_nothing_flag, 
                                                enable_waiting_flag)
        current_job_probability = assign_result_matrix[-1] # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
        choose_indexes = []
        
        current_job_probability_list = list(current_job_probability)
        waiting_sort_indexes = range(sign_matrix.shape[1])
        current_job_probability_sorted_indexes = sorted(waiting_sort_indexes, key=lambda k: (datablock_privacy_budget_remain_list[k], current_job_probability_list[k]), reverse=True)
        
        for sorted_index in current_job_probability_sorted_indexes:
            if sorted_index in choose_indexes:
                continue
            prob_true = min(1.0, max(0.0, current_job_probability_list[sorted_index]))
            if sub_train_datasetidentifier_2_epsilon_remain[temp_index_2_datablock_identifier[sorted_index]] < target_epsilon_require:
                prob_true = 0.0
            if prob_true > 0.0:
                if self.is_greedy_flag and prob_true > self.greedy_threshold:
                    choose_indexes.append(sorted_index)
                    self.logger.debug(f"(job_id[{job_id}], datablock_identifier[{temp_index_2_datablock_identifier[sorted_index]}]) => remain: {datablock_privacy_budget_remain_list[sorted_index]}; pro: {current_job_probability_list[sorted_index]}")
                else:
                    prob_false = 1.0 - prob_true
                    prob_vec = [prob_false, prob_true]
                    choice_result = np.random.choice(a=range(2), size=1, replace=False, p=prob_vec)
                    if choice_result == 1:
                        choose_indexes.append(sorted_index)
                    self.logger.debug(f"(job_id[{job_id}], datablock_identifier[{temp_index_2_datablock_identifier[sorted_index]}]) => remain: {datablock_privacy_budget_remain_list[sorted_index]}; pro: {current_job_probability_list[sorted_index]}; choice_result: {choice_result}")
        
        self.logger.debug(f"job_id[{job_id}] step[pro]: choose_indexes: {choose_indexes}")

        for choose_index in choose_indexes:
            datablock_identifier = temp_index_2_datablock_identifier[choose_index]
            if target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                selected_datablock_identifiers.append(datablock_identifier)
                selected_real_sched_epsilon_map[(job_id, datablock_identifier)] = target_epsilon_require
        return selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon

    def get_allocation(self, state, all_or_nothing_flag, enable_waiting_flag):
        job_id, train_dataset_name = self.get_allocation_judge_one_job(state)
        self.add_to_policy_profiler(job_id)

        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
        sub_train_datasetidentifier_2_arrival_time = state["current_sub_train_datasetidentifier_2_arrival_time"][train_dataset_name]
        
        target_epsilon_require = state["job_id_2_target_epsilon_require"][job_id]
        target_datablock_select_num = state["job_id_2_target_datablock_selected_num"][job_id]
        target_arrival_time = state["job_id_2_arrival_time"][job_id]
        job_priority_weight = state["job_id_2_job_priority_weight"][job_id]
        sub_train_datasetidentifier_2_significance = state["job_id_2_significance"][job_id]
        job_arrival_index = state["job_id_2_arrival_index"][job_id]
        job_test_dataset_name = state["job_id_2_test_dataset_name"][job_id]
        job_sub_test_key_id = state["job_id_2_sub_test_key_id"][job_id]
        job_train_dataset_name = state["job_id_2_train_dataset_name"][job_id]
        job_model_name = state["job_id_2_model_name"][job_id]
        
        all_job_sequence_num = self.job_request_all_num # TODO(xlc): 需要all_job_seq_num
        sample_history_and_current_job_request_all_num = self.sample_history_and_current_job_request_all_num
        offline_history_job_ids = self.offline_history_job_ids
        offline_history_job_priority_weights = self.offline_history_job_priority_weights
        offline_history_job_budget_consumes = self.offline_history_job_budget_consumes
        offline_history_job_signficance = self.offline_history_job_significance
        offline_history_job_target_datablock_selected_num = self.offline_history_job_target_selected_num
        offline_history_job_arrival_time = self.offline_history_job_arrival_time
        offline_history_job_test_dataset_name = self.offline_history_job_test_dataset_name
        offline_history_job_sub_test_key_id = self.offline_history_job_sub_test_key_id
        offline_history_job_train_dataset_name = self.offline_history_job_train_dataset_name
        offline_history_job_model_name = self.offline_history_job_model_name

        online_history_job_ids = self.online_history_job_ids
        online_history_job_priority_weights = self.online_history_job_priority_weights
        online_history_job_budget_consumes = self.online_history_job_budget_consumes
        online_history_job_signficance = self.online_history_job_significance
        online_history_job_target_datablock_selected_num = self.online_history_job_target_selected_num
        online_history_job_arrival_time = self.online_history_job_arrival_time
        online_history_job_test_dataset_name = self.online_history_job_test_dataset_name
        online_history_job_sub_test_key_id = self.online_history_job_sub_test_key_id
        online_history_job_train_dataset_name = self.online_history_job_train_dataset_name
        online_history_job_model_name = self.online_history_job_model_name

        # assert target_datablock_select_num == 1
        
        if self.adaptive_n_flag:
            # offline sample per round
            if self.adaptive_offline_h_num < len(offline_history_job_ids):
                offline_sample_indexes = np.random.choice(range(len(offline_history_job_ids)), self.adaptive_offline_h_num, replace=False)
            else:
                offline_sample_indexes = range(len(offline_history_job_ids))
            online_sample_indexes = range(len(online_history_job_ids))
        else:
            if len(offline_history_job_priority_weights) + len(online_history_job_priority_weights) < sample_history_and_current_job_request_all_num:
                offline_sample_indexes = range(len(offline_history_job_ids))
                online_sample_indexes = range(len(online_history_job_ids))
            else:
                select_num_from_offline_history = max(sample_history_and_current_job_request_all_num - len(online_history_job_priority_weights) - 1, 0)
                offline_sample_indexes = np.random.choice(range(len(offline_history_job_priority_weights)), select_num_from_offline_history, replace=False)
                
                if len(online_history_job_priority_weights) > sample_history_and_current_job_request_all_num - 1:
                    online_sample_indexes = np.random.choice(range(len(online_history_job_priority_weights)), self.pipeline_sequence_all_num - 1, replace=False)
                else:
                    online_sample_indexes = range(len(online_history_job_priority_weights))
        self.logger.debug(f"offline_sample_indexes: {offline_sample_indexes}; online_sample_indexes: {online_sample_indexes}")
        sample_history_job_ids = [offline_history_job_ids[i] for i in offline_sample_indexes] + [online_history_job_ids[i] for i in online_sample_indexes]
        sample_history_job_priority_weights = [offline_history_job_priority_weights[i] for i in offline_sample_indexes] + [online_history_job_priority_weights[i] for i in online_sample_indexes] 
        sample_history_job_budget_consumes = [offline_history_job_budget_consumes[i] for i in offline_sample_indexes] + [online_history_job_budget_consumes[i] for i in online_sample_indexes]
        sample_history_job_signficances = [offline_history_job_signficance[i] for i in offline_sample_indexes] + [online_history_job_signficance[i] for i in online_sample_indexes]
        sample_history_job_target_datablock_selected_nums = [offline_history_job_target_datablock_selected_num[i] for i in offline_sample_indexes] + [online_history_job_target_datablock_selected_num[i] for i in online_sample_indexes]
        sample_history_job_arrival_times = [offline_history_job_arrival_time[i] for i in offline_sample_indexes] + [online_history_job_arrival_time[i] for i in online_sample_indexes]
        sample_history_job_test_dataset_names = [offline_history_job_test_dataset_name[i] for i in offline_sample_indexes] + [online_history_job_test_dataset_name[i] for i in online_sample_indexes]
        sample_history_job_sub_test_key_ids = [offline_history_job_sub_test_key_id[i] for i in offline_sample_indexes] + [online_history_job_sub_test_key_id[i] for i in online_sample_indexes]
        sample_history_job_train_dataset_names = [offline_history_job_train_dataset_name[i] for i in offline_sample_indexes] + [online_history_job_train_dataset_name[i] for i in online_sample_indexes]
        sample_history_job_model_names = [offline_history_job_model_name[i] for i in offline_sample_indexes] + [online_history_job_model_name[i] for i in online_sample_indexes]
        if (not self.is_infinity_flag) and job_arrival_index < self.beta * all_job_sequence_num: # TODO(xlc): 无限任务的时候, beta只能设置为0
            self.logger.info("stop due to sample caused by job_arrival_index: {}; self.beta: {}; all_job_sequence_num: {}".format(
                job_arrival_index, self.beta, all_job_sequence_num
            ))
            temp_selected_datablock_identifiers = []
            temp_selected_real_sched_epsilon_map = {}
            calcu_compare_epsilon = 0.0
        else:
            temp_selected_datablock_identifiers, temp_selected_real_sched_epsilon_map, \
                calcu_compare_epsilon = self.get_allocation_for_small(job_id, 
                                sample_history_job_ids,
                                sample_history_job_priority_weights, 
                                sample_history_job_budget_consumes, 
                                sample_history_job_signficances, 
                                sample_history_job_target_datablock_selected_nums,
                                sample_history_job_arrival_times,
                                sample_history_job_test_dataset_names,
                                sample_history_job_sub_test_key_ids,
                                sample_history_job_train_dataset_names,
                                sample_history_job_model_names,
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                sub_train_datasetidentifier_2_arrival_time,
                                target_epsilon_require, 
                                target_datablock_select_num, 
                                target_arrival_time,
                                job_priority_weight,
                                job_test_dataset_name,
                                job_sub_test_key_id,
                                job_train_dataset_name,
                                job_model_name,
                                all_or_nothing_flag, 
                                enable_waiting_flag)
   
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
        
        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name , result_job_2_selected_datablock_identifiers))
        return result_job_2_selected_datablock_identifiers, result_waiting_job_ids, result_selected_real_sched_epsilon_map, calcu_compare_epsilon, result_job_2_instant_recoming_flag

    def adaptive_calculate_adaptive_h(self):
        all_history_job_budget_consumes = self.offline_history_job_budget_consumes
        all_history_job_target_datablock_selected_nums = self.offline_history_job_target_selected_num
        adaptive_offline_h_num = len(all_history_job_budget_consumes)
        if self.adaptive_n_flag:
            _, all_blocks_require_mean = self.get_mean_require(all_history_job_budget_consumes, all_history_job_target_datablock_selected_nums)
            if all_blocks_require_mean > 0.0:
                max_need_operator_job_num = int(self.datablocks_privacy_budget_all / all_blocks_require_mean) - 1 if int(self.datablocks_privacy_budget_all / all_blocks_require_mean) > 1 else 0
            else:
                max_need_operator_job_num = len(all_history_job_budget_consumes)
            if max_need_operator_job_num < len(all_history_job_budget_consumes):
                adaptive_offline_h_num = max_need_operator_job_num
        
        self.logger.info(f"update datablocks_privacy_budget_all: {self.datablocks_privacy_budget_all}")
        self.logger.info(f"update adaptive_offline_h_num: {adaptive_offline_h_num}")
        self.adaptive_offline_h_num = adaptive_offline_h_num