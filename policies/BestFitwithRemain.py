from policies.BasePolicy import Policy


class BestFitwithRemainPolicy(Policy):
    def __init__(self, seed, logger):
        super().__init__()
        self._name = 'BestFitwithRemainPolicy'
        self.logger = logger
        self.waiting_queue_capacity = 1

        self.only_one = True
        self.need_history = False

    def report_state(self):
        self.logger.info("policy name: {}".format(self._name))
        self.logger.info("policy args: None")    
    
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
        
        
        temp_datasetidentifier_2_epsilon_z = {}
        for datasetidentifier in sub_train_datasetidentifier_2_epsilon_remain:
            if sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier] >= target_epsilon_require:
                temp_datasetidentifier_2_epsilon_z[datasetidentifier] = sub_train_datasetidentifier_2_epsilon_remain[datasetidentifier] / sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
        
        count = 0
        calcu_compare_epsilon = 0.0
        selected_datablock_identifiers = []
        selected_real_sched_epsilon_map = {}
        enable_sub_train_datasetidentifier_2_significance = {key: value for key, value in temp_datasetidentifier_2_epsilon_z.items()}
        sorted_sub_train_datasetidentifier_2_significance = sorted(enable_sub_train_datasetidentifier_2_significance.items(), key=lambda x: x[1], reverse=True)
        while count < target_datablock_select_num and len(temp_datasetidentifier_2_epsilon_z.keys()) > 0:
            # 获取价值最高的数据块
            # datasetidentifier = random.choice(list(temp_datasetidentifier_2_epsilon_z.keys()))
            datasetidentifier, significance = sorted_sub_train_datasetidentifier_2_significance.pop(0)
            datablock_epsilon_capacity = sub_train_datasetidentifier_2_epsilon_capcity[datasetidentifier]
            datablock_z = temp_datasetidentifier_2_epsilon_z[datasetidentifier]            
            selected_datablock_identifiers.append(datasetidentifier)
            selected_real_sched_epsilon_map[(job_id, datasetidentifier)] = target_epsilon_require
            # final_datasetidentifier_2_epsilon_z[datasetidentifier] = new_z
            del temp_datasetidentifier_2_epsilon_z[datasetidentifier]
            count += 1

        job_2_selected_datablock_identifiers = [
            (job_id, identifier) for identifier in selected_datablock_identifiers
        ]
        
        self.logger.debug("from policy [{}] selected_datablock_identifiers: {}".format(self.name , job_2_selected_datablock_identifiers))
        return job_2_selected_datablock_identifiers, selected_real_sched_epsilon_map, calcu_compare_epsilon