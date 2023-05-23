class Policy:
    def __init__(self, job_sequence_all_num):
        self._name = None
        self._job_sequence_all_num = job_sequence_all_num
        self.has_come_job_ids = set()

    @property
    def name(self):
        return self._name

    @property
    def job_sequence_all_num(self):
        return self._job_sequence_all_num

    @property
    def all_job_has_coming(self):
        return len(self.has_come_job_ids) == self.job_sequence_all_num

    def add_to_policy_profiler(self, job_id):
        self.has_come_job_ids.add(job_id)

    def exist_in_policy_profiler(self, job_id):
        return job_id in self.has_come_job_ids

    def get_allocation_judge_one_job(self, state):
        job_id_2_train_dataset_name = state["job_id_2_train_dataset_name"]
        assert len(job_id_2_train_dataset_name) == 1
        set_job_id = set(job_id_2_train_dataset_name.keys())
        set_dataset_name = set(job_id_2_train_dataset_name.values())
        assert len(set_dataset_name) == 1 # 必须保证所有的任务都是针对同一个数据集的
        job_id = list(set_job_id)[0]
        train_dataset_name = list(set_dataset_name)[0]
        return job_id, train_dataset_name

    
    def push_success_allocation(self, success_datasetidentifier_2_consume_epsilon):
        pass

    def push_offline_history_to_assignment_policy(self, offline_history_job_priority_weights, offline_history_job_budget_consumes,
        offline_history_job_target_selected_num, offline_history_job_train_dataset_name, offline_history_job_test_dataset_name,
        offline_history_job_sub_test_key_id, offline_history_job_type_id, offline_history_job_significance):
        pass

    def push_online_history_to_assignment_policy(self, online_job_priority_weight, online_job_budget_consume, 
        online_job_datablock_selected_num, online_job_train_dataset_name, online_job_test_dataset_name, 
        online_job_sub_test_key_id, online_job_type_id, online_job_significance):
        pass

    def pull_offline_history_from_assignment_policy(self, target_keys):
        pass

    def pull_online_history_from_assignment_policy(self, target_keys):
        pass

    def update_offline_history_job_significance_to_assignment_policy(self, offline_history_job_significance):
        pass
    
    def update_online_history_job_significance_to_assignment_policy(self, online_history_job_significance):
        pass
    