from policies.BasePolicy import Policy
import copy
import random
import numpy as np
import math
import cvxpy as cp
import json
import sys

class HISBasePolicy(Policy):
    def __init__(self, beta, pipeline_sequence_all_num, job_request_all_num, 
                infinity_flag, 
                greedy_flag, greedy_threshold,
                seed, logger):
        super().__init__(pipeline_sequence_all_num, job_request_all_num)
        self.beta = beta
        self.logger = logger
        self.waiting_queue_capacity = 1
        self.only_one = True
        self.need_history = True
        self._infinity_flag = infinity_flag
        self._greedy_flag = greedy_flag
        self._greedy_threshold = greedy_threshold
        self.initialize_seeds(seed)
    
        self.offline_history_job_ids = []
        self.offline_history_job_priority_weights = []
        self.offline_history_job_budget_consumes = []
        self.offline_history_job_target_selected_num = []
        self.offline_history_job_type_id = []
        self.offline_history_job_significance = []
        self.offline_history_job_arrival_time = []
        self.offline_history_job_test_dataset_name = []
        self.offline_history_job_sub_test_key_id = []
        self.offline_history_job_train_dataset_name = []
        self.offline_history_job_model_name = []

        self.online_history_job_ids = []
        self.online_history_job_priority_weights = []
        self.online_history_job_budget_consumes = []
        self.online_history_job_target_selected_num = []
        self.online_history_job_type_id = []
        self.online_history_job_significance = []
        self.online_history_job_arrival_time = []
        self.online_history_job_test_dataset_name = []
        self.online_history_job_sub_test_key_id = []
        self.online_history_job_train_dataset_name = []
        self.online_history_job_model_name = []

    @property
    def sample_history_and_current_job_request_all_num(self):
        if self.is_infinity_flag:
            return sys.maxsize
        else:
            return self.job_request_all_num

    @property
    def is_greedy_flag(self):
        return self._greedy_flag

    @property
    def greedy_threshold(self):
        if self._greedy_flag:
            return self._greedy_threshold
        else:
            return 1.0

    @property
    def is_infinity_flag(self):
        return self._infinity_flag or self.job_request_all_num == sys.maxsize
    
    def initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)
    
    def push_offline_history_to_assignment_policy(self, offline_history_job_ids,
            offline_history_job_priority_weights, offline_history_job_budget_consumes,
            offline_history_job_target_selected_num, offline_history_job_train_dataset_name, offline_history_job_test_dataset_name,
            offline_history_job_sub_test_key_id, offline_history_job_type_id, offline_history_job_significance, 
            offline_history_job_arrival_time, offline_history_job_model_name):
        self.offline_history_job_ids = offline_history_job_ids
        self.offline_history_job_priority_weights = offline_history_job_priority_weights
        self.offline_history_job_budget_consumes = offline_history_job_budget_consumes
        self.offline_history_job_target_selected_num = offline_history_job_target_selected_num    
        self.offline_history_job_type_id = offline_history_job_type_id
        self.offline_history_job_significance = offline_history_job_significance
        self.offline_history_job_arrival_time = offline_history_job_arrival_time
        self.offline_history_job_test_dataset_name = offline_history_job_test_dataset_name
        self.offline_history_job_sub_test_key_id = offline_history_job_sub_test_key_id
        self.offline_history_job_train_dataset_name = offline_history_job_train_dataset_name
        self.offline_history_job_model_name = offline_history_job_model_name

    def push_online_history_to_assignment_policy(self, online_history_job_id, 
            online_job_priority_weight, online_job_budget_consume, 
            online_job_datablock_selected_num, online_job_train_dataset_name, online_job_test_dataset_name, 
            online_job_sub_test_key_id, online_job_type_id, online_job_significance, 
            online_job_arrival_time, online_job_model_name):
        self.online_history_job_ids.append(online_history_job_id)
        self.online_history_job_priority_weights.append(online_job_priority_weight)
        self.online_history_job_budget_consumes.append(online_job_budget_consume)
        self.online_history_job_target_selected_num.append(online_job_datablock_selected_num)
        self.online_history_job_type_id.append(online_job_type_id)
        self.online_history_job_significance.append(online_job_significance)
        self.online_history_job_arrival_time.append(online_job_arrival_time)
        self.online_history_job_test_dataset_name.append(online_job_test_dataset_name)
        self.online_history_job_sub_test_key_id.append(online_job_sub_test_key_id)
        self.online_history_job_train_dataset_name.append(online_job_train_dataset_name)
        self.online_history_job_model_name.append(online_job_model_name)

    def pull_offline_history_from_assignment_policy(self, target_keys):
        result = {}
        for key in target_keys:
            if key == "offline_history_job_ids":
                result[key] = self.offline_history_job_ids
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
            if key == "offline_history_job_arrival_time":
                result[key] = self.offline_history_job_arrival_time
            if key == "offline_history_job_model_name":
                result[key] = self.offline_history_job_model_name
        return result

    def pull_online_history_from_assignment_policy(self, target_keys):
        result = {}
        for key in target_keys:
            if key == "online_history_job_ids":
                result[key] = self.online_history_job_ids
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
            if key == "online_history_job_arrival_time":
                result[key] = self.online_history_job_arrival_time
            if key == "online_history_job_model_name":
                result[key] = self.online_history_job_model_name
        return result

    def update_offline_history_job_significance_to_assignment_policy(self, offline_history_job_significance):
        self.offline_history_job_significance = offline_history_job_significance
    
    def update_online_history_job_significance_to_assignment_policy(self, online_history_job_significance):
        self.online_history_job_significance = online_history_job_significance

    def get_mean_require(self, current_all_job_budget_consumes, current_all_job_target_datablock_selected_nums):
        current_all_job_budget_consumes_np = np.array(current_all_job_budget_consumes)
        current_all_job_target_datablock_selected_nums_np = np.array(current_all_job_target_datablock_selected_nums)
        one_block_require_mean = np.sum(current_all_job_budget_consumes_np) / len(current_all_job_budget_consumes)
        all_blocks_require_mean = np.sum(current_all_job_budget_consumes_np * current_all_job_target_datablock_selected_nums_np) / len(current_all_job_budget_consumes)
        self.logger.debug(f"one_block_require_mean: {one_block_require_mean}")
        self.logger.debug(f"all_blocks_require_mean: {all_blocks_require_mean}")
        return one_block_require_mean, all_blocks_require_mean

    def get_his_single_job_require_epsilon(self, current_all_job_budget_consumes, current_all_job_target_datablock_selected_nums, batch_size_for_one_epoch):
        one_block_require_mean, all_blocks_require_mean = self.get_mean_require(current_all_job_budget_consumes, current_all_job_target_datablock_selected_nums)
        need_siton_block_num_mean = int(batch_size_for_one_epoch * all_blocks_require_mean / one_block_require_mean)
        self.logger.debug(f"need_siton_block_num_mean: {need_siton_block_num_mean}")
        return one_block_require_mean, all_blocks_require_mean, need_siton_block_num_mean

    def get_his_right_capacity_for_single_job_Fair(self, current_all_job_budget_consumes,
                                                target_epsilon_require,
                                                current_all_job_target_datablock_selected_nums,
                                                target_datablock_select_num,
                                                datablock_privacy_budget_remain_list, 
                                                datablock_privacy_budget_capacity_list,
                                                batch_size_for_one_epoch):
        one_block_require_mean, all_blocks_require_mean, need_siton_block_num_mean = self.get_his_single_job_require_epsilon(
            current_all_job_budget_consumes, current_all_job_target_datablock_selected_nums, batch_size_for_one_epoch
        )
        siton_block_epsilon_capacity = one_block_require_mean
        
        valid_remain_siton_num_per_block = np.floor_divide(datablock_privacy_budget_remain_list, siton_block_epsilon_capacity)
        valid_remain_siton_all_blocks = np.sum(valid_remain_siton_num_per_block)
        
        result = np.zeros_like(datablock_privacy_budget_remain_list)
        current_require = need_siton_block_num_mean
        if valid_remain_siton_all_blocks < current_require:
            # 需要全部开放
            result = valid_remain_siton_num_per_block
            current_require -= np.sum(valid_remain_siton_num_per_block)
        else:
            # 每次获取非0的index, 均匀取一个, 直到满足需求或者全部非0
            non_zero_valid_remain_siton_num_per_block_index = np.nonzero(valid_remain_siton_num_per_block)[0]
            while current_require > 0 and len(non_zero_valid_remain_siton_num_per_block_index) > 0:
                if len(non_zero_valid_remain_siton_num_per_block_index) > current_require:
                    # 需要根据当前数量进行采样
                    non_zero_valid_remain_sition_num_per_block_value = valid_remain_siton_num_per_block[non_zero_valid_remain_siton_num_per_block_index]
                    non_zero_valid_remain_sition_num_per_block_value = non_zero_valid_remain_sition_num_per_block_value / np.sum(non_zero_valid_remain_sition_num_per_block_value)
                    
                    samples = np.random.choice(non_zero_valid_remain_siton_num_per_block_index, size=int(current_require), p=non_zero_valid_remain_sition_num_per_block_value)
                else:
                    samples = non_zero_valid_remain_siton_num_per_block_index
                result[samples] += 1
                valid_remain_siton_num_per_block[samples] -= 1
                current_require -= len(samples)
                non_zero_valid_remain_siton_num_per_block_index = np.nonzero(valid_remain_siton_num_per_block)[0]
        self.logger.debug(f"valid_remain_siton_num_per_block: {valid_remain_siton_num_per_block}")
        self.logger.debug(f"first open from remain result: {result}")
        self.logger.debug(f"first current_require: {current_require}")

        datablock_privacy_budget_used_list = np.subtract(datablock_privacy_budget_capacity_list, datablock_privacy_budget_remain_list)
        valid_used_siton_num_per_block = np.floor_divide(datablock_privacy_budget_used_list, siton_block_epsilon_capacity)
        valid_used_siton_all_blocks = np.sum(valid_used_siton_num_per_block)

        if valid_used_siton_all_blocks < current_require:
            result += valid_used_siton_num_per_block
            current_require -= np.sum(valid_used_siton_num_per_block)
        else:
            non_zero_valid_used_siton_num_per_block_index = np.nonzero(valid_used_siton_num_per_block)[0]
            while current_require > 0 and len(non_zero_valid_used_siton_num_per_block_index) > 0:
                if len(non_zero_valid_used_siton_num_per_block_index) > current_require:
                    samples = np.random.choice(non_zero_valid_used_siton_num_per_block_index, size=int(current_require))
                else:
                    samples = non_zero_valid_used_siton_num_per_block_index
                result[samples] += 1
                valid_used_siton_num_per_block[samples] -= 1
                current_require -= len(samples)
                non_zero_valid_used_siton_num_per_block_index = np.nonzero(valid_used_siton_num_per_block)[0]

        self.logger.debug(f"valid_used_siton_num_per_block: {valid_used_siton_num_per_block}")
        self.logger.debug(f"second open from used result: {result}")
        self.logger.debug(f"second current_require: {current_require}")

        right_capacity_for_single_job = siton_block_epsilon_capacity * result
        self.logger.debug(f"first right_capacity_for_single_job: {right_capacity_for_single_job}")

        result_zero_index = np.where(result == 0)[0]
        for index in result_zero_index:
            if datablock_privacy_budget_remain_list[index] >= target_epsilon_require and datablock_privacy_budget_remain_list[index] <= siton_block_epsilon_capacity:
                right_capacity_for_single_job[index] = datablock_privacy_budget_remain_list[index]

        self.logger.debug(f"second right_capacity_for_single_job: {right_capacity_for_single_job}")
        self.logger.debug(f"sum(right_capacity_for_single_job): {np.sum(right_capacity_for_single_job)}")
        self.logger.debug(f"this batch jobs consume: {np.sum(np.multiply(current_all_job_budget_consumes, current_all_job_target_datablock_selected_nums))}")
        return right_capacity_for_single_job

    def get_his_right_capacity_MinBlock(self, current_all_job_budget_consumes,
                                                target_epsilon_require,
                                                current_all_job_target_datablock_selected_nums,
                                                target_datablock_select_num,
                                                datablock_privacy_budget_remain_list, 
                                                datablock_privacy_budget_capacity_list,
                                                batch_size_for_one_epoch):
        _, all_blocks_require_mean, _ = self.get_his_single_job_require_epsilon(
            current_all_job_budget_consumes, current_all_job_target_datablock_selected_nums, batch_size_for_one_epoch
        )
        datablock_supply_mean = np.mean(datablock_privacy_budget_capacity_list)
        psi = len(current_all_job_budget_consumes)
        
        self.logger.debug(f"psi: {psi}; all_blocks_require_mean: {all_blocks_require_mean}; => np.ceil(psi * all_blocks_require_mean / datablock_supply_mean): {np.ceil(psi * all_blocks_require_mean / datablock_supply_mean)} => len(datablock_privacy_budget_capacity_list): {len(datablock_privacy_budget_capacity_list)}")
        to_open_datablock_num = min(int(np.ceil(psi * all_blocks_require_mean / datablock_supply_mean)), len(datablock_privacy_budget_capacity_list))

        right_capacity_for_single_job = [0.0] * len(datablock_privacy_budget_capacity_list)
        sorted_remain_budget_with_index = sorted(enumerate(datablock_privacy_budget_remain_list), key=lambda x: x[1], reverse=True)
        for origin_index, remain_budget in sorted_remain_budget_with_index[0:to_open_datablock_num]:
            # self.logger.debug(f"open origin_index: {origin_index} => remain_budget: {remain_budget}")
            right_capacity_for_single_job[origin_index] = datablock_privacy_budget_capacity_list[origin_index]
        # self.logger.debug(f"to_open_datablock_num: {to_open_datablock_num}")
        return right_capacity_for_single_job