from policies.BasePolicy import Policy
import copy
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

class OfflinePolicy(Policy):
    def __init__(self, job_sequence_all_num, seed, logger):
        super().__init__()
        self._name = 'OfflinePolicy'
        # 保存一个unlocked的量
        self.waiting_queue = []
        self.waiting_queue_jobid_set = set()
        self.waiting_queue_capacity = job_sequence_all_num

        self.only_one = False
        self.need_history = False

        self.logger = logger  
    
    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
    
    def get_allocation(self, state):
        job_id_2_target_epsilon_require = state["job_id_2_target_epsilon_require"]
        job_id_2_target_datablock_select_num = state["job_id_2_target_datablock_selected_num"]
        job_id_2_job_priority_weight = state["job_id_2_job_priority_weight"]
        job_id_2_train_dataset_name = state["job_id_2_train_dataset_name"]
        job_id_2_significance = state["job_id_2_significance"]
        # all_job_sequence_num = state["all_job_sequence_num"]

        set_dataset_name = set(job_id_2_train_dataset_name.values())
        assert len(set_dataset_name) == 1 # 必须保证所有的任务都是针对同一个数据集的
        train_dataset_name = list(set_dataset_name)[0]

        # sub_train_datasetidentifier_2_significance = state["current_sub_train_datasetidentifier_2_significance"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
    
        for job_id, train_dataset_name in job_id_2_train_dataset_name.items():
            if job_id not in self.waiting_queue_jobid_set:
                target_datablock_select_num = job_id_2_target_datablock_select_num[job_id]
                target_epsilon_require = job_id_2_target_epsilon_require[job_id]
                job_priority_weight = job_id_2_job_priority_weight[job_id]
                sub_train_datasetidentifier_2_significance = job_id_2_significance[job_id]
                waiting_job = WaitingJob(job_id, train_dataset_name, target_datablock_select_num, target_epsilon_require, job_priority_weight, sub_train_datasetidentifier_2_significance)
                self.waiting_queue_jobid_set.add(job_id)
                self.waiting_queue.append(waiting_job)
        
        job_2_selected_datablock_identifiers = {} 
        selected_real_sched_epsilon_map = {}
        calcu_compare_epsilon = 0.0
        waiting_job_ids = []
        
        if len(self.waiting_queue) >= self.waiting_queue_capacity:
            job_2_selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon = self.on_scheduler_time(
                sub_train_datasetidentifier_2_epsilon_remain,
                sub_train_datasetidentifier_2_epsilon_capcity
            )
            self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name , job_2_selected_datablock_identifiers))
            self.waiting_queue = []
        else:
            waiting_job_ids = [job_item.job_id for job_item in self.waiting_queue]
        return job_2_selected_datablock_identifiers, waiting_job_ids, selected_real_sched_epsilon_map, calcu_compare_epsilon

    def on_scheduler_time(self, 
                        sub_train_datasetidentifier_2_epsilon_remain,
                        sub_train_datasetidentifier_2_epsilon_capcity):
        job_2_selected_datablock_identifiers, \
            selected_real_sched_epsilon_map, \
            calcu_compare_epsilon = self.get_allocation_for_small(sub_train_datasetidentifier_2_epsilon_remain,
                                                                sub_train_datasetidentifier_2_epsilon_capcity)
        return job_2_selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon

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
        job_target_datablock_selected_num_list = np.array(job_target_datablock_selected_num_list)[np.newaxis, :]
        job_privacy_budget_consume_list = np.array(job_privacy_budget_consume_list)[np.newaxis, :]
        datablock_privacy_budget_capacity_list = np.array(datablock_privacy_budget_capacity_list)[np.newaxis, :]

        matrix_X = cp.Variable((job_num, datablock_num), nonneg=True)
        objective = cp.Maximize(
            cp.sum(cp.multiply(sign_matrix, matrix_X))
        )

        # self.logger.debug(f"========= multiply ========")
        # self.logger.debug(np.sum(np.multiply(sign_matrix, matrix_X)))
        # self.logger.debug(f"========= sign_matrix.value [{sign_matrix.shape}]=========")
        # self.logger.debug(sign_matrix)
        # self.logger.debug(f"========= job_target_datablock_selected_num_list.value [{job_target_datablock_selected_num_list.shape}] [{np.squeeze(job_target_datablock_selected_num_list).shape}]=========")
        # self.logger.debug(job_target_datablock_selected_num_list)
        # self.logger.debug(f"========= job_privacy_budget_consume_list.value [{job_privacy_budget_consume_list.shape}]=========")
        # self.logger.debug(job_privacy_budget_consume_list)
        # self.logger.debug(f"========= datablock_privacy_budget_capacity_list.value [{datablock_privacy_budget_capacity_list.shape}]=========")
        # self.logger.debug(datablock_privacy_budget_capacity_list)
        # self.logger.debug(f"========= job_privacy_budget_consume_list @ matrix_X [{(job_privacy_budget_consume_list @ matrix_X).shape}]=========")

        constraints = [
            matrix_X >= 0,
            matrix_X <= 1,
            cp.sum(matrix_X, axis=1) <= np.squeeze(job_target_datablock_selected_num_list),
            (job_privacy_budget_consume_list @ matrix_X) <= datablock_privacy_budget_capacity_list
        ]

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver, verbose=False)
        # print(matrix_X.value)
        if cvxprob.status != "optimal":
            self.logger.warning('WARNING: Allocation returned by policy not optimal!')

        self.logger.debug("========= matrix_X.value =========")
        self.logger.debug(matrix_X.value)
        if matrix_X.value is None:
            result = np.zeros(shape=(job_num, datablock_num))
        else:
            result = matrix_X.value
        return result
    
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

        for job_item in self.waiting_queue:
            current_all_job_priority_weights.append(job_item.job_priority_weight)
            current_all_job_budget_consumes.append(job_item.target_epsilon_require)
            current_all_job_significances.append(job_item.sub_train_datasetidentifier_2_significance)
            current_all_job_target_datablock_selected_nums.append(job_item.target_datablock_select_num)
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
        
        job_2_selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        
        pairs = []
        for job_row_index in range(job_num):
            for datablock_col_index in range(datablock_num):
                pairs.append((job_row_index, datablock_col_index, assign_result_matrix[job_row_index][datablock_col_index]))
        sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        job_current_selected_datablock_count = [0] * job_num
        for pair in sorted_pairs:
            job_index, datablock_index = pair[0], pair[1]
            job_item = self.waiting_queue[job_index]
            datablock_identifier = temp_index_2_datablock_identifier[datablock_index]
            
            if job_current_selected_datablock_count[job_index] < job_item.target_datablock_select_num:
                job_2_selected_datablock_identifiers.append((job_item.job_id, datablock_identifier))
                selected_real_sched_epsilon_map[(job_item.job_id, datablock_identifier)] = job_item.target_epsilon_require
                job_current_selected_datablock_count[job_index] += 1
    
        return job_2_selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon