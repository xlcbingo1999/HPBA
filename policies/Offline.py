from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
import cvxpy as cp
import heapq
import json

class WaitingJob(object):
    def __init__(self, job_id, train_dataset_name, target_datablock_select_num, target_epsilon_require, job_priority_weight, sub_train_datasetidentifier_2_significance):
        self.job_id = job_id
        self.train_dataset_name = train_dataset_name
        self.target_datablock_select_num = target_datablock_select_num
        self.target_epsilon_require = target_epsilon_require
        self.job_priority_weight = job_priority_weight
        self.sub_train_datasetidentifier_2_significance = sub_train_datasetidentifier_2_significance
        self.dominant_share = 0.0

        self.only_one = False
        self.need_history = False

class OfflinePolicy(Policy):
    def __init__(self, waiting_queue_capacity, logger):
        super().__init__()
        self._name = 'OfflinePolicy'
        # 保存一个unlocked的量
        self.waiting_queue = []
        self.waiting_queue_capacity = waiting_queue_capacity

        self.logger = logger  
        self.initialize_seeds(1234)
    
    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: waiting_queue_capacity: {}".format(self.waiting_queue_capacity))

    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

    def get_allocation(self, state):
        job_id_2_target_epsilon_require = state["job_id_2_target_epsilon_require"]
        job_id_2_target_datablock_select_num = state["job_id_2_target_datablock_selected_num"]
        job_id_2_job_priority_weight = state["job_id_2_job_priority_weight"]
        job_id_2_train_dataset_name = state["job_id_2_train_dataset_name"]
        job_id_2_significance = state["job_id_2_significance"]
        # all_job_sequence_num = state["all_job_sequence_num"]

        assert len(job_id_2_train_dataset_name) >= self.waiting_queue_capacity
        set_dataset_name = set(job_id_2_train_dataset_name.values())
        assert len(set_dataset_name) == 1 # 必须保证所有的任务都是针对同一个数据集的
        train_dataset_name = list(set_dataset_name)[0]

        # sub_train_datasetidentifier_2_significance = state["current_sub_train_datasetidentifier_2_significance"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
    
        for job_id, train_dataset_name in job_id_2_train_dataset_name.items():
            target_datablock_select_num = job_id_2_target_datablock_select_num[job_id]
            target_epsilon_require = job_id_2_target_epsilon_require[job_id]
            job_priority_weight = job_id_2_job_priority_weight[job_id]
            sub_train_datasetidentifier_2_significance = job_id_2_significance[job_id]
            waiting_job = WaitingJob(job_id, train_dataset_name, target_datablock_select_num, target_epsilon_require, job_priority_weight, sub_train_datasetidentifier_2_significance)
            self.waiting_queue.append(waiting_job)
        job_2_selected_datablock_identifiers = {} 
        calcu_compare_epsilon = 0.0
            # 根据dominant_share排序等待任务
        job_2_selected_datablock_identifiers, calcu_compare_epsilon = self.on_scheduler_time(
            sub_train_datasetidentifier_2_epsilon_remain,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        self.waiting_queue = []
        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name , job_2_selected_datablock_identifiers))
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon

    def on_scheduler_time(self, 
                        sub_train_datasetidentifier_2_epsilon_remain,
                        sub_train_datasetidentifier_2_epsilon_capcity):
        job_2_selected_datablock_identifiers, \
            calcu_compare_epsilon = self.get_allocation_for_small(sub_train_datasetidentifier_2_epsilon_remain,
                                                                sub_train_datasetidentifier_2_epsilon_capcity)
        return job_2_selected_datablock_identifiers, calcu_compare_epsilon

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

    def get_LP_result(self, sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, job_privacy_budget_consume_list, solver=cp.ECOS_BB):
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
        datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

        matrix_X = cp.Variable((job_num, datablock_num), integer=True)
        objective = cp.Maximize(
            cp.sum(cp.multiply(sign_matrix, matrix_X))
        )

        constraints = [
            matrix_X >= 0,
            matrix_X <= 1,
            cp.sum(matrix_X, axis=1) <= job_target_datablock_selected_num_list,
            (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
        ]

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver)
        # print(matrix_X.value)
        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')
        return matrix_X.value
    
    def get_schedule_order(self, waiting_job_selected_sign):
        job_num = len(waiting_job_selected_sign)
        temp_sign_list = [0] * job_num
        heap = []
        for i in range(job_num):
            temp_sign_list[i] = len(waiting_job_selected_sign[i])
            if temp_sign_list[i] > 0:
                for j in range(temp_sign_list[i]):
                    heapq.heappush(heap, (-waiting_job_selected_sign[i][j], (i, j)))
        schedule_order = []
        while len(heap) > 0:
            _, (x, y) = heapq.heappop(heap)
            schedule_order.append((x, y))
        return schedule_order

    def get_allocation_for_small(self, sub_train_datasetidentifier_2_epsilon_remain, sub_train_datasetidentifier_2_epsilon_capcity):
        # assert target_datablock_select_num == 1
        calcu_compare_epsilon = 0.0
        
        current_all_job_priority_weights = []
        current_all_job_budget_consumes = []
        current_all_job_significances = []
        current_all_job_target_datablock_selected_nums = []

        for job in self.waiting_queue:
            current_all_job_priority_weights.append(job.job_priority_weight)
            current_all_job_budget_consumes.append(job.target_epsilon_require)
            current_all_job_significances.append(job.sub_train_datasetidentifier_2_significance)
            current_all_job_target_datablock_selected_nums.append(job.target_datablock_select_num)
        sign_matrix, temp_index_2_datablock_identifier = self.get_sign_matrix(
            current_all_job_priority_weights,
            current_all_job_significances,
            sub_train_datasetidentifier_2_epsilon_capcity
        )
        
        datablock_privacy_budget_capacity_list = np.zeros(shape=sign_matrix.shape[1])
        job_target_datablock_selected_num_list = np.array(current_all_job_target_datablock_selected_nums)
        for temp_index in temp_index_2_datablock_identifier:
            datablock_privacy_budget_capacity_list[temp_index] = sub_train_datasetidentifier_2_epsilon_capcity[temp_index_2_datablock_identifier[temp_index]]
        assign_result_matrix = self.get_LP_result(sign_matrix, datablock_privacy_budget_capacity_list, job_target_datablock_selected_num_list, current_all_job_budget_consumes)
        job_num, datablock_num = sign_matrix.shape[0], sign_matrix.shape[1]
        waiting_job_selected_datablock_identifiers = []
        waiting_job_selected_sign = []
        for index, job in enumerate(self.waiting_queue):
            temp_selected_datablock_identifiers = []
            temp_selected_probability = []

            current_job_probability = list(assign_result_matrix[-(len(self.waiting_queue) - index)]) # 这里其实相当于算出了一个分数, 如果为了这个分数不被泄露, 可以用指数机制加噪, 该方案被证实为满足DP-差分隐私.
            
            result_select_num = job.target_datablock_select_num
            temp_result = heapq.nlargest(result_select_num, range(len(current_job_probability)), current_job_probability.__getitem__)
            choose_indexes = [index for index in temp_result if current_job_probability[index]]
            for choose_index in choose_indexes:
                datablock_identifier = temp_index_2_datablock_identifier[choose_index]
                temp_selected_datablock_identifiers.append(datablock_identifier)
                temp_selected_probability.append(current_job_probability[choose_index])
            waiting_job_selected_datablock_identifiers.append(temp_selected_datablock_identifiers)
            waiting_job_selected_sign.append(temp_selected_probability)
        scheduler_order = self.get_schedule_order(waiting_job_selected_sign)
        job_2_selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        for x, y in scheduler_order:
            job = self.waiting_queue[x]
            datablock_identifier = waiting_job_selected_datablock_identifiers[x][y]
            if job.target_epsilon_require <= sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier]:
                job_2_selected_datablock_identifiers.append((job.job_id, datablock_identifier))
                selected_real_sched_epsilon_map[(job.job_id, datablock_identifier)] = job.target_epsilon_require
                sub_train_datasetidentifier_2_epsilon_remain[datablock_identifier] -= job.target_epsilon_require

        return job_2_selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon