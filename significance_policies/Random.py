from significance_policies.BaseSigPolicy import SigPolicy
import json
from utils.global_variable import DATASET_PATH, SIGNIFICANCE_TRACE_PREFIX_PATH
from utils.data_loader import get_concat_dataset
import time
import os
import random
from tqdm import tqdm

class RandomPolicy(SigPolicy):
    def __init__(self, dataset_config_name, simulation, logger):
        super().__init__()
        self._name = "RandomPolicy"
        
        
        self.dataset_config_name = dataset_config_name
        if simulation:
            self.Random_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_RandomPolicy_{dataset_config_name}.json"
        else:
            self.Random_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_RandomPolicy_{dataset_config_name}.json"
        
        if os.path.exists(self.Random_trace_path):
            with open(self.Random_trace_path, "r+") as f:
                self.origin_Random_trace = json.load(f)
        else:
            self.origin_Random_trace = {}
    
        self.logger = logger

    def set_origin_Random_trace_value(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name, target_value):
        if train_dataset_name not in self.origin_Random_trace:
            self.origin_Random_trace[train_dataset_name] = {}
        if sub_train_key_id not in self.origin_Random_trace[train_dataset_name]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.origin_Random_trace[train_dataset_name][sub_train_key_id]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        if sub_test_key_id not in self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = {}
        self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id][model_name] = target_value
    
    def get_job_datablock_origin_Random_sync(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name):
        if train_dataset_name not in self.origin_Random_trace:
            self.origin_Random_trace[train_dataset_name] = {}
        if sub_train_key_id not in self.origin_Random_trace[train_dataset_name]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.origin_Random_trace[train_dataset_name][sub_train_key_id]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        if sub_test_key_id not in self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = {}
        if model_name not in self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id]:
            self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id][model_name] = random.random()

        result_d = self.origin_Random_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id][model_name]
        return result_d

    def get_job_significance_result_for_all_datablocks(self, all_significance_state):
        begin = time.time()
        significance_origin_Temps_map = {}
        significance_norm_Temps_map = {}
        for job_id in all_significance_state:
            for datablock_identifier in all_significance_state[job_id]:
                signficance_state = all_significance_state[job_id][datablock_identifier]
                train_dataset_name = signficance_state["train_dataset_name"]
                sub_train_key_id = signficance_state["sub_train_key_id"]
                test_dataset_name = signficance_state["test_dataset_name"]
                sub_test_key_id = signficance_state["sub_test_key_id"]
                model_name = signficance_state["model_name"]

                origin_Random_d = self.get_job_datablock_origin_Random_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name)
                significance_origin_Temps_map.setdefault(job_id, {})[datablock_identifier] = origin_Random_d
        for job_id in all_significance_state:
            for datablock_identifier in significance_origin_Temps_map[job_id]:
                significance_norm_Temps_map.setdefault(job_id, {})[datablock_identifier] = significance_origin_Temps_map[job_id][datablock_identifier]
            
        
        end = time.time()
        self.logger.debug("norm_Randoms: {}, time: {}".format(
            significance_norm_Temps_map, end-begin
        ))
        return significance_norm_Temps_map
'''    
if __name__ == "__main__":
    import itertools
    train_datablock_num = 10000
    dataset_name = "EMNIST"
    dataset_config_name = f"subtrain_{train_datablock_num}_split_1.0_dirichlet"
    type_id = "No_importance"
    train_dataset_names = ["EMNIST"]
    sub_train_key_ids = [f"train_sub_{index}" for index in range(train_datablock_num)]
    test_dataset_names = ["EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"] # "EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"
    sub_test_key_ids = ["test_sub_0"]
    model_names = ["FF", "CNN"]
    
    DATASET_PATH = "/mnt/linuxidc_client/dataset"
    SIGNIFICANCE_TRACE_PREFIX_PATH = DATASET_PATH + "/traces"
    Random_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_RandomPolicy_{dataset_config_name}.json"
    
    origin_Random_trace = {}
    for train_dataset_name in train_dataset_names:
        for sub_train_key_id in tqdm(sub_train_key_ids):
            for test_dataset_name in test_dataset_names:
                for sub_test_key_id in sub_test_key_ids:
                    for model_name in model_names:
                        origin_Random_trace.setdefault(train_dataset_name, {}).setdefault(sub_train_key_id, {}).setdefault(test_dataset_name, {}).setdefault(sub_test_key_id, {})[model_name] = random.random()
    with open(Random_trace_path, "w+") as f:
        json.dump(origin_Random_trace, f)
'''