class Policy:

    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name
    
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
    