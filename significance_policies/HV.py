# 这个函数没有什么问题
from significance_policies.BaseSigPolicy import SigPolicy
import time
        

class JobTypeItem(object): # TODO: 改成 JobTypeItem
    def __init__(self, type_identifier, train_dataset_name, test_dataset_name, sub_test_key_id):
        self.type_identifier = type_identifier
        self.type_train_dataset_name = train_dataset_name
        self.type_test_dataset_name = test_dataset_name
        self.type_sub_test_key_id = sub_test_key_id

        self.history_acces = []
        self.history_delta_acces = []
        self.history_used_subtrain_block_per_epoches = []
        self.history_subtrain_block_used_count = {}
        
        self.history_significance_for_subtrain_datablock = {}

    def get_history_result(self, sub_train_key_id, history_rho):
        max_index = len(self.history_delta_acces) - 1
        history_result_fenzi = 0.0
        history_result_fenmu = 0.0
        # print("job {} first in {}".format(self.type_id, len(self.history_delta_acces)))
        for index, delta_acc in enumerate(self.history_delta_acces):
            add_delta_acc_value = 0.0
            if sub_train_key_id in self.history_used_subtrain_block_per_epoches[index]:
                add_delta_acc_value = delta_acc
            weight_rho = pow(history_rho, max_index - index)
            history_result_fenzi += weight_rho * add_delta_acc_value
            history_result_fenmu += weight_rho
        return history_result_fenzi / history_result_fenmu

class HVPolicy(SigPolicy):
    def __init__(self, logger, batch_size=16, history_alpha=0.5, history_rho=0.5):
        super().__init__()
        self._name = "HVPolicy"
        self.distance_batch_size = batch_size
        self.calculate_batch_size = batch_size
        self.history_rho = history_rho

        self.logger = logger

        self.need_update_backward = True

        self.type_identifier_2_typeitem = {}

    def get_job_significance_result_for_history_jobs_for_all_datablocks(self, history_type_id, all_significance_state):
        begin = time.time()
        result = [
            1.0 for index in range(len(all_significance_state))
        ]
        end = time.time()
        self.logger.info("history_type_id [{}] to datablocks significance: {}, time: {}".format(history_type_id, result, end-begin))
        return result

    def get_job_significance_result_for_all_datablocks(self, type_id, all_significance_state):
        begin = time.time()

        origin_history_accs = []

        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            # sub_train_key_remain_epsilon = signficance_state["sub_train_key_remain_epsilon"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]

            # 获取原始的history_acc
            if type_id not in self.type_identifier_2_typeitem:
                self.type_identifier_2_typeitem[type_id] = JobTypeItem(type_id, train_dataset_name, test_dataset_name, sub_test_key_id)
            
            if len(self.type_identifier_2_typeitem[type_id].history_acces) > 0:
                history_result = self.type_identifier_2_typeitem[type_id].get_history_result(sub_train_key_id, self.history_rho)
            else:
                history_result = 0.0
            origin_history_accs.append(history_result)

        norm_history_accs = []
        temp_max_history_acc = max(origin_history_accs)
        temp_min_history_acc = min(origin_history_accs)
        if temp_max_history_acc != temp_min_history_acc:
            for origin_acc in origin_history_accs:
                norm_acc = (origin_acc - temp_min_history_acc) / (temp_max_history_acc - temp_min_history_acc) # TODO(xlc): 这个方法会让结果直接降到0
                norm_history_accs.append(norm_acc)
        else:
            for origin_acc in origin_history_accs:
                norm_acc = 1.0
                norm_history_accs.append(norm_acc)
        
            
        # 全局量 * (局部量 + UCB), 对量纲的影响是最小的
        # 不能把当前时刻的remain_epsilon传进来, 会导致历史任务的价值偏高, 当前任务的价值不断下降
        # 太久没选的任务是否要将探索价值提高呢? 如果在世界时间中, 当最后的任务价值不断提高, 也会导致历史任务的价值不断提高...
        # 实际上很大概率就是任务在第一次被failed后, 整个系统会将其拒之门外...
        result = [
            norm_history_accs[index] for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.info("type_id [{}] to datablocks significance: {} [norm_history_accs: {}], time: {}".format(
            type_id, result, norm_history_accs, end-begin
        ))
        return result

    def get_job_datablock_significance_async(self, type_id, signficance_state, device_index, is_history):
        pass
        
    def update_job_datablock_signficance_FAIR(self, type_id, used_sub_train_key_ids, current_result):
        # 需要有一些真实的结果来调整重要性评估指标, 将每一次计算得到的delta_loss反馈回场景中
        # 因为所有维度都固定, 所以delta_loss的变化幅度不会非常大?
        current_acc = current_result["test_acc"]
        if type_id not in self.type_identifier_2_typeitem:
            raise ValueError("type_id must in self.type_identifier_2_typeitem")
        type_item = self.type_identifier_2_typeitem[type_id]
        if len(type_item.history_acces) == 0:
            type_item.history_delta_acces.append(current_acc - 0.0)
        else:
            type_item.history_delta_acces.append(current_acc - type_item.history_acces[-1])
        type_item.history_acces.append(current_acc)
        type_item.history_used_subtrain_block_per_epoches.append(used_sub_train_key_ids)
        for sub_train_key in used_sub_train_key_ids:
            if sub_train_key not in type_item.history_subtrain_block_used_count:
                type_item.history_subtrain_block_used_count[sub_train_key] = 0
            type_item.history_subtrain_block_used_count[sub_train_key] += 1