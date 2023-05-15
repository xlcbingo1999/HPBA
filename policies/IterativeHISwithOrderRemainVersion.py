from policies.BasePolicy import Policy
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

class IterativeHISwithOrderRemainVersionPolicy(Policy):
    def __init__(self, beta, job_sequence_all_num, batch_size_for_one_epoch, seed, logger):
        super().__init__()
        self._name = 'IterativeHISwithOrderRemainVersionPolicy'
        self.beta = beta
        self.z0 = 0.1
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

        self.online_history_job_priority_weights = []
        self.online_history_job_budget_consumes = []
        self.online_history_job_target_selected_num = []
        self.online_history_job_train_dataset_name = []
        self.online_history_job_test_dataset_name = []
        self.online_history_job_sub_test_key_id = []
        self.online_history_job_type_id = []
        self.online_history_job_significance = []
    
    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: beta: {}".format(self.beta))
        self.logger.info("policy args: z0: {}".format(self.z0))
        self.logger.info("policy args: all_epoch_num: {}".format(self.all_epoch_num))
        self.logger.info("policy args: batch_size_for_one_epoch: {}".format(self.batch_size_for_one_epoch))
        # self.logger.info("policy args: delta: {}".format(self.delta))
        # self.logger.info("policy args: only_small: {}".format(self.only_small))

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

    def get_LP_result(self, sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, job_privacy_budget_consume_list, solver=cp.ECOS):
        begin_time = time.time()
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        self.logger.debug("job_num: {}; datablock_num: {}".format(job_num, datablock_num))
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

        # self.logger.debug("check job_target_datablock_selected_num_list: {}".format(job_target_datablock_selected_num_list))
        # self.logger.debug("check datablock_privacy_budget_capacity_list: {}".format(datablock_privacy_budget_capacity_list))
        # self.logger.debug("check sum of job_privacy_budget_consume_list: {}".format(np.sum(job_privacy_budget_consume_list * job_target_datablock_selected_num_list)))

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
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                target_epsilon_require, 
                                target_datablock_select_num, 
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

        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            current_all_job_priority_weights,
            current_all_job_signficances,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        datablock_privacy_budget_remain_list = np.zeros(shape=sign_matrix.shape[1])
        job_target_datablock_selected_num_list = np.array(current_all_job_target_datablock_selected_nums)

        for temp_index in temp_index_2_datablock_identifier:
            datablock_privacy_budget_capacity_list[temp_index] = self.datablock_identifier_2_remain_epsilon[temp_index_2_datablock_identifier[temp_index]] + \
                sub_train_datasetidentifier_2_epsilon_capcity[temp_index_2_datablock_identifier[temp_index]] / self.datablock_identifier_2_all_epoch_num[temp_index_2_datablock_identifier[temp_index]]
        
        for temp_index, datablock_identifier in temp_index_2_datablock_identifier.items():
            datablock_privacy_budget_remain_list[temp_index] = sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]

        assign_result_matrix = self.get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, current_all_job_budget_consumes)
        current_job_probability = assign_result_matrix[-1] # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
        choose_indexes = []
        
        current_job_probability_list = list(current_job_probability)
        waiting_sort_indexes = range(sign_matrix.shape[1])
        current_job_probability_sorted_indexes = sorted(waiting_sort_indexes, key=lambda k: (datablock_privacy_budget_remain_list[k], current_job_probability_list[k]), reverse=True)
        
        for sorted_index in current_job_probability_sorted_indexes:
            if sorted_index in choose_indexes:
                continue
            prob_true = min(1.0, max(0.0, current_job_probability_list[sorted_index]))
            prob_false = 1.0 - prob_true
            prob_vec = [prob_false, prob_true]
            choice_result = np.random.choice(a=range(2), size=1, replace=False, p=prob_vec)
            if choice_result == 1:
                choose_indexes.append(sorted_index)

            self.logger.debug(f"(job_id[{job_id}], datablock_identifier[{temp_index_2_datablock_identifier[sorted_index]}]) => remain: {datablock_privacy_budget_remain_list[sorted_index]}; pro: {current_job_probability_list[sorted_index]}; choice_result: {choice_result}")

        # best-effort
        '''
        self.logger.debug(f"remain03 === z0: {(self.current_batch_size_for_one_epoch + 1) / self.batch_size_for_one_epoch}")
        z_bigger_than_z0_indexes = []
        current_z = []
        sub_job_probability_list = []
        for temp_index in temp_index_2_datablock_identifier:
            if target_epsilon_require <= ((self.current_batch_size_for_one_epoch + 1) / self.batch_size_for_one_epoch) * self.datablock_identifier_2_remain_epsilon[temp_index_2_datablock_identifier[temp_index]]:
                current_z.append(target_epsilon_require / self.datablock_identifier_2_remain_epsilon[temp_index_2_datablock_identifier[temp_index]])
                z_bigger_than_z0_indexes.append(temp_index)
                sub_job_probability_list.append(current_job_probability_list[temp_index])
        self.logger.debug(f"z_bigger_than_z0_indexes: {z_bigger_than_z0_indexes}")
        '''
        '''
        self.logger.debug(f"remain02 === z0: {job_arrival_index / all_job_sequence_num}")
        z_bigger_than_z0_indexes = []
        current_z = []
        sub_job_probability_list = []
        for temp_index in temp_index_2_datablock_identifier:
            if target_epsilon_require <= (job_arrival_index / all_job_sequence_num) * datablock_privacy_budget_remain_list[temp_index]:
                current_z.append(target_epsilon_require / datablock_privacy_budget_remain_list[temp_index])
                z_bigger_than_z0_indexes.append(temp_index)
                sub_job_probability_list.append(current_job_probability_list[temp_index])
        self.logger.debug(f"z_bigger_than_z0_indexes: {z_bigger_than_z0_indexes}")
        '''
        self.logger.debug(f"remain01 === z0: {self.z0}")
        z_bigger_than_z0_indexes = []
        current_z = []
        sub_job_probability_list = []
        for temp_index in temp_index_2_datablock_identifier:
            if target_epsilon_require <= self.z0 * datablock_privacy_budget_remain_list[temp_index]:
                current_z.append(target_epsilon_require / datablock_privacy_budget_remain_list[temp_index])
                z_bigger_than_z0_indexes.append(temp_index)
                sub_job_probability_list.append(current_job_probability_list[temp_index])
        self.logger.debug(f"z_bigger_than_z0_indexes: {z_bigger_than_z0_indexes}")


        assert len(current_z) == len(z_bigger_than_z0_indexes) == len(sub_job_probability_list)
        current_z_sorted_secondary_indexes = sorted(range(len(z_bigger_than_z0_indexes)), key=lambda k: (sub_job_probability_list[k], current_z[k]), reverse=True)
        for temp_secondary_index in current_z_sorted_secondary_indexes:
            z_bigger_than_z0_index = z_bigger_than_z0_indexes[temp_secondary_index]
            if z_bigger_than_z0_index not in choose_indexes:
                choose_indexes.append(z_bigger_than_z0_index)
                self.logger.debug(f"job_id[{job_id}] add datablock identifier caused by z0: {temp_index_2_datablock_identifier[z_bigger_than_z0_index]}")

        self.logger.debug(f"job_id[{job_id}] step[pro and z0]: choose_indexes: {choose_indexes}")

        for choose_index in choose_indexes:
            datablock_identifier = temp_index_2_datablock_identifier[choose_index]
            if len(selected_datablock_identifiers) < target_datablock_select_num and target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                selected_datablock_identifiers.append(datablock_identifier)
                selected_real_sched_epsilon_map[(job_id, datablock_identifier)] = target_epsilon_require
        return selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon

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
        
        all_job_sequence_num = self.job_sequence_all_num
        offline_history_job_priority_weights = self.offline_history_job_priority_weights
        offline_history_job_budget_consumes = self.offline_history_job_budget_consumes
        offline_history_job_signficance = self.offline_history_job_significance
        offline_history_job_target_datablock_selected_num = self.offline_history_job_target_selected_num

        online_history_job_priority_weights = self.online_history_job_priority_weights
        online_history_job_budget_consumes = self.online_history_job_budget_consumes
        online_history_job_signficance = self.online_history_job_significance
        online_history_job_target_datablock_selected_num = self.online_history_job_target_selected_num
        
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

        if job_arrival_index % self.batch_size_for_one_epoch < self.beta * self.batch_size_for_one_epoch:
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
                                sub_train_datasetidentifier_2_significance,
                                sub_train_datasetidentifier_2_epsilon_remain, 
                                sub_train_datasetidentifier_2_epsilon_capcity,
                                target_epsilon_require, target_datablock_select_num, job_priority_weight,
                                job_arrival_index, all_job_sequence_num)
   
        job_2_selected_datablock_identifiers = [
            (job_id, identifier) for identifier in selected_datablock_identifiers
        ]
        waiting_job_ids = []
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
        
    def push_offline_history_to_assignment_policy(self, offline_history_job_priority_weights, offline_history_job_budget_consumes,
            offline_history_job_target_selected_num, offline_history_job_train_dataset_name, offline_history_job_test_dataset_name,
            offline_history_job_sub_test_key_id, offline_history_job_type_id, offline_history_job_significance):
        self.offline_history_job_priority_weights = offline_history_job_priority_weights
        self.offline_history_job_budget_consumes = offline_history_job_budget_consumes
        self.offline_history_job_target_selected_num = offline_history_job_target_selected_num
        self.offline_history_job_train_dataset_name = offline_history_job_train_dataset_name
        self.offline_history_job_test_dataset_name = offline_history_job_test_dataset_name
        self.offline_history_job_sub_test_key_id = offline_history_job_sub_test_key_id
        self.offline_history_job_type_id = offline_history_job_type_id
        self.offline_history_job_significance = offline_history_job_significance

    def push_online_history_to_assignment_policy(self, online_job_priority_weight, online_job_budget_consume, 
            online_job_datablock_selected_num, online_job_train_dataset_name, online_job_test_dataset_name, 
            online_job_sub_test_key_id, online_job_type_id, online_job_significance):
        self.online_history_job_priority_weights.append(online_job_priority_weight)
        self.online_history_job_budget_consumes.append(online_job_budget_consume)
        self.online_history_job_target_selected_num.append(online_job_datablock_selected_num)
        self.online_history_job_train_dataset_name.append(online_job_train_dataset_name)
        self.online_history_job_test_dataset_name.append(online_job_test_dataset_name)
        self.online_history_job_sub_test_key_id.append(online_job_sub_test_key_id)
        self.online_history_job_type_id.append(online_job_type_id)
        self.online_history_job_significance.append(online_job_significance)

    def pull_offline_history_from_assignment_policy(self, target_keys):
        result = {}
        for key in target_keys:
            if key == "offline_history_job_priority_weights":
                result[key] = self.offline_history_job_priority_weights
            if key == "offline_history_job_budget_consumes":
                result[key] = self.offline_history_job_budget_consumes
            if key == "offline_history_job_target_selected_num":
                result[key] = self.offline_history_job_target_selected_num
            if key == "offline_history_job_train_dataset_name":
                result[key] = self.offline_history_job_train_dataset_name
            if key == "offline_history_job_test_dataset_name":
                result[key] = self.offline_history_job_test_dataset_name
            if key == "offline_history_job_sub_test_key_id":
                result[key] = self.offline_history_job_sub_test_key_id
            if key == "offline_history_job_type_id":
                result[key] = self.offline_history_job_type_id
            if key == "offline_history_job_significance":
                result[key] = self.offline_history_job_significance
        return result

    def pull_online_history_from_assignment_policy(self, target_keys):
        result = {}
        for key in target_keys:
            if key == "online_history_job_priority_weights":
                result[key] = self.online_history_job_priority_weights
            if key == "online_history_job_budget_consumes":
                result[key] = self.online_history_job_budget_consumes
            if key == "online_history_job_target_selected_num":
                result[key] = self.online_history_job_target_selected_num
            if key == "online_history_job_train_dataset_name":
                result[key] = self.online_history_job_train_dataset_name
            if key == "online_history_job_test_dataset_name":
                result[key] = self.online_history_job_test_dataset_name
            if key == "online_history_job_sub_test_key_id":
                result[key] = self.online_history_job_sub_test_key_id
            if key == "online_history_job_type_id":
                result[key] = self.online_history_job_type_id
            if key == "online_history_job_significance":
                result[key] = self.online_history_job_significance
        return result

    def update_offline_history_job_significance_to_assignment_policy(self, offline_history_job_significance):
        self.offline_history_job_significance = offline_history_job_significance
    
    def update_online_history_job_significance_to_assignment_policy(self, online_history_job_significance):
        self.online_history_job_significance = online_history_job_significance
    