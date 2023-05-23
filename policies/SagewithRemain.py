from policies.BasePolicy import Policy
import random


class SagewithRemainPolicy(Policy):
    def __init__(self, job_sequence_all_num, seed, logger):
        super().__init__(job_sequence_all_num)
        self._name = 'SagewithRemainPolicy'
        self.logger = logger
        self.waiting_queue_capacity = 1

        self.only_one = True
        self.need_history = False
        self.initialize_seeds(seed)

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: None")    

    def initialize_seeds(self, seed):
        random.seed(seed+1)
    
    def get_allocation(self, state, all_or_nothing_flag, enable_waiting_flag):
        need_waiting_job_sched = False
        job_id, train_dataset_name = self.get_allocation_judge_one_job(state)
        self.add_to_policy_profiler(job_id)
        
        sub_train_datasetidentifier_2_epsilon_remain = state["current_sub_train_datasetidentifier_2_epsilon_remain"][train_dataset_name]
        sub_train_datasetidentifier_2_epsilon_capcity = state["current_sub_train_datasetidentifier_2_epsilon_capcity"][train_dataset_name]
        target_epsilon_require = state["job_id_2_target_epsilon_require"][job_id]
        target_datablock_select_num = state["job_id_2_target_datablock_selected_num"][job_id]
        job_priority_weight = state["job_id_2_job_priority_weight"][job_id]
        sub_train_datasetidentifier_2_significance = state["job_id_2_significance"][job_id]
        
        temp_datasetidentifier_2_epsilon_z = {}
        for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain:
            if sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier] >= target_epsilon_require:
                temp_datasetidentifier_2_epsilon_z[datasetidentifier] = sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier] / sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
        
        count = 0
        calcu_compare_epsilon = 0.0
        temp_selected_datablock_identifiers = []
        temp_selected_real_sched_epsilon_map = {}
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取随机一个数据集
            datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]            
            temp_selected_datablock_identifiers.append(datasetidentifier)
            temp_selected_real_sched_epsilon_map[(job_id, datasetidentifier)] = target_epsilon_require
            # final_datasetidentifier_2_epsilon_z[datasetidentifier] = new_z
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
            count += 1

        result_job_2_selected_datablock_identifiers = {}
        result_selected_real_sched_epsilon_map = {}
        temp_sched_failed_flag = False
        if ((not all_or_nothing_flag) and len(temp_selected_datablock_identifiers) > 0) or (all_or_nothing_flag and len(temp_selected_datablock_identifiers) == target_datablock_select_num):
            result_job_2_selected_datablock_identifiers[job_id] = temp_selected_datablock_identifiers
            result_selected_real_sched_epsilon_map = temp_selected_real_sched_epsilon_map
        else:
            temp_sched_failed_flag = True
        waiting_job_ids = []
        if enable_waiting_flag:
            need_waiting_job_sched = need_waiting_job_sched or False
            if temp_sched_failed_flag:
                waiting_job_ids.append(job_id)

        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name, result_job_2_selected_datablock_identifiers))
        return result_job_2_selected_datablock_identifiers, waiting_job_ids, result_selected_real_sched_epsilon_map, calcu_compare_epsilon, need_waiting_job_sched