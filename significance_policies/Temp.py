from significance_policies.BaseSigPolicy import SigPolicy
import json
import time

from utils.global_variable import SIGNIFICANCE_TRACE_PREFIX_PATH

class TempPolicy(SigPolicy):
    def __init__(self, simulation, logger):
        super().__init__()
        self._name = "TempPolicy"

        self.need_update_backward = False

        if simulation:
            self.significance_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + "/significance_TempPolicy_2000.json"
        else:
            self.significance_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + "/significance_TempPolicy.json"
        self.logger = logger
        with open(self.significance_trace_path, "r+") as f:
            self.significance_trace = json.load(f)
        # self.logger.info("self.significance_trace: {}".format(self.significance_trace))

    def value_in_origin_temp_trace(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id):
        if train_dataset_name not in self.significance_trace:
            return None
        if sub_train_key_id not in self.significance_trace[train_dataset_name]:
            return None
        if test_dataset_name not in self.significance_trace[train_dataset_name][sub_train_key_id]:
            return None
        if sub_test_key_id not in self.significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            return None
        return self.significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id]


    def get_job_significance_result_for_history_jobs_for_all_datablocks(self, history_type_id, all_significance_state):
        begin = time.time()
        origin_Temps = []
        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]

            # 获取原始的temp
            origin_temp_d = self.value_in_origin_temp_trace(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
            origin_Temps.append(origin_temp_d)
        
        result = [
            origin_Temps[index] for index in range(len(all_significance_state))
        ]
        end = time.time()
        # self.logger.info("history_type_id [{}] to datablocks significance: {}, time: {}".format(history_type_id, result, end-begin))
        return result

    def get_job_significance_result_for_all_datablocks(self, type_id, all_significance_state):
        begin = time.time()
        origin_Temps = []
        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]

            # 获取epsilon的剩余值
            # remain_epsilons.append(sub_train_key_remain_epsilon)

            # 获取原始的temp
            origin_temp_d = self.value_in_origin_temp_trace(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
            origin_Temps.append(origin_temp_d)
            
        # 全局量 * (局部量 + UCB), 对量纲的影响是最小的
        # 不能把当前时刻的remain_epsilon传进来, 会导致历史任务的价值偏高, 当前任务的价值不断下降
        # 太久没选的任务是否要将探索价值提高呢? 如果在世界时间中, 当最后的任务价值不断提高, 也会导致历史任务的价值不断提高...
        # 实际上很大概率就是任务在第一次被failed后, 整个系统会将其拒之门外...
        result = [
            origin_Temps[index] for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.debug("type_id [{}] to datablocks significance: {} [origin_Temps: {}], time: {}".format(
            type_id, result, origin_Temps, end-begin
        ))
        return result

        