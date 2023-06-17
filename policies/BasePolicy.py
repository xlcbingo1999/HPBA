import sys
import numpy as np

class Policy:
    def __init__(self, pipeline_sequence_all_num, job_request_all_num):
        self._name = None
        self._pipeline_sequence_all_num = pipeline_sequence_all_num
        self._job_request_all_num = job_request_all_num
        self.has_come_pipeline_ids = set()

        self.backward_cal_significance_func = None

    def register_backward_cal_significance_func(self, backward_cal_significance_func):
        self.backward_cal_significance_func = backward_cal_significance_func

    @property
    def name(self):
        return self._name

    @property
    def pipeline_sequence_all_num(self):
        return self._pipeline_sequence_all_num

    @property
    def job_request_all_num(self):
        return self._job_request_all_num

    @property
    def all_pipeline_has_coming(self):
        return len(self.has_come_pipeline_ids) == self.pipeline_sequence_all_num

    def add_to_policy_profiler(self, job_id):
        self.has_come_pipeline_ids.add(job_id)

    def exist_in_policy_profiler(self, job_id):
        return job_id in self.has_come_pipeline_ids

    def get_allocation_judge_one_job(self, state):
        job_id_2_train_dataset_name = state["job_id_2_train_dataset_name"]
        assert len(job_id_2_train_dataset_name) == 1
        set_job_id = set(job_id_2_train_dataset_name.keys())
        set_dataset_name = set(job_id_2_train_dataset_name.values())
        assert len(set_dataset_name) == 1 # 必须保证所有的任务都是针对同一个数据集的
        job_id = list(set_job_id)[0]
        train_dataset_name = list(set_dataset_name)[0]
        return job_id, train_dataset_name

    def get_sign_matrix(self, current_all_job_ids, 
                        current_all_job_priority_weights, 
                        current_all_job_significances,
                        current_all_job_test_dataset_names,
                        current_all_job_sub_test_key_ids,
                        current_all_job_train_dataset_names,
                        current_all_job_model_names,
                        sub_train_datasetidentifier_2_epsilon_capcity):
        temp_index_2_datablock_identifier = {}
        sign_matrix = []
        for job_index, job_priority_weight in enumerate(current_all_job_priority_weights):
            temp = []
            for datablock_index, datablock_identifier in enumerate(sub_train_datasetidentifier_2_epsilon_capcity):
                temp_index_2_datablock_identifier[datablock_index] = datablock_identifier
                if datablock_identifier not in current_all_job_significances[job_index]:
                    self.logger.warning(f"[warning] pair({job_index}, {datablock_identifier}) not in current_all_job_significances, try to cal")
                    job_id = current_all_job_ids[job_index]
                    test_dataset_name = current_all_job_test_dataset_names[job_index]
                    sub_test_key_id = current_all_job_sub_test_key_ids[job_index]
                    train_dataset_name = current_all_job_train_dataset_names[job_index]
                    sub_train_key_id = datablock_identifier
                    model_name = current_all_job_model_names[job_index]
                    significance_state = {}
                    significance_state.setdefault(job_id, {}).setdefault(datablock_identifier, {})["test_dataset_name"] = test_dataset_name
                    significance_state.setdefault(job_id, {}).setdefault(datablock_identifier, {})["sub_test_key_id"] = sub_test_key_id
                    significance_state.setdefault(job_id, {}).setdefault(datablock_identifier, {})["train_dataset_name"] = train_dataset_name
                    significance_state.setdefault(job_id, {}).setdefault(datablock_identifier, {})["sub_train_key_id"] = sub_train_key_id
                    significance_state.setdefault(job_id, {}).setdefault(datablock_identifier, {})["model_name"] = model_name
                    result_d_map = self.backward_cal_significance_func(
                        f"recal job_id({job_id}) datablock({datablock_identifier}) significance in get_sign_matrix",
                        significance_state
                    )
                    sig = result_d_map[job_id][datablock_identifier] * job_priority_weight
                else:
                    sig = current_all_job_significances[job_index][datablock_identifier] * job_priority_weight
                temp.append(sig)
            sign_matrix.append(temp)
        sign_matrix = np.array(sign_matrix)
        return sign_matrix, temp_index_2_datablock_identifier

    
    def push_success_allocation(self, success_datasetidentifier_2_consume_epsilon):
        pass

    def push_offline_history_to_assignment_policy(self, offline_history_job_ids, 
        offline_history_job_priority_weights, offline_history_job_budget_consumes,
        offline_history_job_target_selected_num, offline_history_job_train_dataset_name, offline_history_job_test_dataset_name,
        offline_history_job_sub_test_key_id, offline_history_job_type_id, offline_history_job_significance, offline_history_job_model_name):
        pass

    def push_online_history_to_assignment_policy(self, online_history_job_ids, 
        online_job_priority_weight, online_job_budget_consume, 
        online_job_datablock_selected_num, online_job_train_dataset_name, online_job_test_dataset_name, 
        online_job_sub_test_key_id, online_job_type_id, online_job_significance, online_job_model_name):
        pass

    def pull_offline_history_from_assignment_policy(self, target_keys):
        pass

    def pull_online_history_from_assignment_policy(self, target_keys):
        pass

    def update_offline_history_job_significance_to_assignment_policy(self, offline_history_job_significance):
        pass
    
    def update_online_history_job_significance_to_assignment_policy(self, online_history_job_significance):
        pass
    