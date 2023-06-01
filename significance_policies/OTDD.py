# 这个函数没有什么问题

from significance_policies.BaseSigPolicy import SigPolicy
import json
from otdd.otdd.pytorch.distance import DatasetDistance, FeatureCost
from utils.global_variable import DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH, SIGNIFICANCE_TRACE_PREFIX_PATH
from utils.data_loader import get_concat_dataset
from torch.utils.data import DataLoader
import time
import torch
from torchvision import models
import os
from utils.global_variable import SIGNIFICANCE_TRACE_PREFIX_PATH, DATASET_CONFIG_NAME
import multiprocessing
from tqdm import tqdm

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

def cal_origin_OTDD(signficance_state, device_index, distance_batch_size, calculate_batch_size):
    print(f"begin: {signficance_state} in device_index {device_index}")
    train_dataset_name = signficance_state["train_dataset_name"]
    test_dataset_name = signficance_state["test_dataset_name"]
    sub_train_key_id = signficance_state["sub_train_key_id"]
    sub_test_key_id = signficance_state["sub_test_key_id"]
    # 耗时操作, 计算OTDD
    train_dataset = get_concat_dataset(train_dataset_name, sub_train_key_id, 
                            DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, 
                            "train")
    test_dataset = get_concat_dataset(test_dataset_name, sub_test_key_id,
                                    DATASET_PATH, TEST_DATASET_CONFIG_PATH,
                                    "test")
    train_loader = DataLoader(train_dataset, batch_size=distance_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=distance_batch_size)

    embedder = models.resnet18(pretrained=True).eval()
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                            src_dim = (3,28,28),
                            tgt_embedding = embedder,
                            tgt_dim = (3,28,28),
                            p = 2,
                            device="cuda:{}".format(device_index))

    dist = DatasetDistance(train_loader, test_loader,
                            inner_ot_method = 'exact',
                            debiased_loss = True,
                            feature_cost = feature_cost,
                            sqrt_method = 'spectral',
                            sqrt_niters=10,
                            precision='single',
                            p = 2, entreg = 1e-1,
                            device="cuda:{}".format(device_index),
                            batch_size=calculate_batch_size)


    origin_otdd = float(dist.distance(maxsamples = 10000).detach().cpu())
    print("====== finished =======")
    return origin_otdd


class OTDDPolicy(SigPolicy):
    def __init__(self, simulation, logger, batch_size=16, history_alpha=0.5, history_rho=0.5):
        super().__init__()
        self._name = "OTDDPolicy"
        self.distance_batch_size = batch_size
        self.calculate_batch_size = batch_size
        self.OTDD_history_alpha = history_alpha
        self.history_rho = history_rho

        self.logger = logger

        self.need_update_backward = False

        self.OTDD_beta = 0.02
        self.max_OTDD = 0.0

        if simulation:
            self.OTDD_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_OTDDPolicy_{DATASET_CONFIG_NAME}.json"
        else:
            self.OTDD_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_OTDDPolicy_{DATASET_CONFIG_NAME}.json"
        
        if os.path.exists(self.OTDD_trace_path):
            with open(self.OTDD_trace_path, "r+") as f:
                self.origin_OTDD_trace = json.load(f)
        else:
            self.origin_OTDD_trace = {}
        
        self.type_identifier_2_typeitem = {}

    def write_to_origin_OTDD_trace(self):
        self.logger.debug("==== write_to_origin_OTDD_trace [origin_OTDD_trace] ====")
        self.logger.info(self.origin_OTDD_trace)
        with open(self.OTDD_trace_path, "w+") as f:
            json.dump(self.origin_OTDD_trace, f)

    def set_origin_OTDD_trace_value(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, target_value):
        if train_dataset_name not in self.origin_OTDD_trace:
            self.origin_OTDD_trace[train_dataset_name] = {}
        if sub_train_key_id not in self.origin_OTDD_trace[train_dataset_name]:
            self.origin_OTDD_trace[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.origin_OTDD_trace[train_dataset_name][sub_train_key_id]:
            self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = target_value
        
    def get_job_datablock_origin_OTDD_sync(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id):
        if train_dataset_name not in self.origin_OTDD_trace:
            self.origin_OTDD_trace[train_dataset_name] = {}
        if sub_train_key_id not in self.origin_OTDD_trace[train_dataset_name]:
            self.origin_OTDD_trace[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.origin_OTDD_trace[train_dataset_name][sub_train_key_id]:
            self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        if sub_test_key_id not in self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            device_index = 0
            signficance_state = {
                "train_dataset_name": train_dataset_name,
                "sub_train_key_id": sub_train_key_id,
                "test_dataset_name": test_dataset_name,
                "sub_test_key_id": sub_test_key_id
            }
            distance_batch_size = self.distance_batch_size
            calculate_batch_size = self.calculate_batch_size
            result_d = cal_origin_OTDD(signficance_state, device_index, distance_batch_size, calculate_batch_size)
            self.set_origin_OTDD_trace_value(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, result_d) 
            self.max_OTDD = max(self.max_OTDD, result_d)
        else:
            result_d = self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id]
        return result_d

    
    def get_job_significance_result_for_all_datablocks(self, type_id, all_significance_state):
        begin = time.time()
        origin_OTDDs = []
        norm_OTDDs = []

        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]

            # 获取epsilon的剩余值
            # remain_epsilons.append(sub_train_key_remain_epsilon)

            # 获取原始的OTDD
            origin_otdd_d = self.get_job_datablock_origin_OTDD_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
            origin_OTDDs.append(origin_otdd_d)
            
        # 全局量
        for origin_otdd in origin_OTDDs:
            norm_otdd = 1.0 / origin_otdd # TODO(xlc): 因为在线场景中的区分度实在不高, 因此为了避免引入新的argue点, 还是选择了直接做除法
            norm_OTDDs.append(norm_otdd)
            
        # 全局量 * (局部量 + UCB), 对量纲的影响是最小的
        # 不能把当前时刻的remain_epsilon传进来, 会导致历史任务的价值偏高, 当前任务的价值不断下降
        # 太久没选的任务是否要将探索价值提高呢? 如果在世界时间中, 当最后的任务价值不断提高, 也会导致历史任务的价值不断提高...
        # 实际上很大概率就是任务在第一次被failed后, 整个系统会将其拒之门外...
        result = [
            norm_OTDDs[index] for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.debug("type_id [{}] to datablocks significance: {} [norm_OTDDs: {}], time: {}".format(
            type_id, result, norm_OTDDs, end-begin
        ))
        return result
    
    def get_job_datablock_significance_async(self, type_id, all_significance_state, cal_device_list):
        assert len(cal_device_list) > 0
        begin = time.time()

        origin_OTDDs = []
        norm_OTDDs = []

        group_size = len(cal_device_list)
        split_all_significance_state = [all_significance_state[i:i+group_size] for i in range(0, len(all_significance_state), group_size)]
        
        for sub_all_significance_state in tqdm(split_all_significance_state):
            with multiprocessing.Pool(processes=len(sub_all_significance_state)) as pool:
                distance_batch_size_list = [self.distance_batch_size] * len(sub_all_significance_state)
                calculate_batch_size_list = [self.calculate_batch_size] * len(sub_all_significance_state)
                args_zip = zip(sub_all_significance_state, cal_device_list, distance_batch_size_list, calculate_batch_size_list)
                origin_otdds = pool.map(cal_origin_OTDD, )
            
            for index, origin_otdd in enumerate(origin_otdds):
                signficance_state = sub_all_significance_state[index]
                train_dataset_name = signficance_state["train_dataset_name"]
                test_dataset_name = signficance_state["test_dataset_name"]
                sub_train_key_id = signficance_state["sub_train_key_id"]
                sub_test_key_id = signficance_state["sub_test_key_id"]
                
                self.set_origin_OTDD_trace_value(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, origin_otdd)    
                self.max_OTDD = max(self.max_OTDD, origin_otdd)
                origin_OTDDs.append(origin_otdd)
                self.logger.debug("result distance => [{}-{}-{}-{}]: {}".format(
                    train_dataset_name, test_dataset_name, sub_train_key_id, sub_test_key_id, origin_otdd
                ))
            
            self.write_to_origin_OTDD_trace()

        for origin_otdd in origin_OTDDs:
            # norm_otdd = math.exp(-1.0 / (origin_otdd * self.OTDD_beta)) / math.exp(-1.0 / (self.max_OTDD * self.OTDD_beta))
            norm_otdd = 1.0 / origin_otdd # TODO(xlc): 因为在线场景中的区分度实在不高, 因此为了避免引入新的argue点, 还是选择了直接做除法
            norm_OTDDs.append(norm_otdd)

        result = [
            norm_OTDDs[index] for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.debug("type_id [{}] to datablocks significance: {} [norm_OTDDs: {}], time: {}".format(
            type_id, result, norm_OTDDs, end-begin
        ))
        return result
