# 这个函数没有什么问题
from significance_policies.BaseSigPolicy import SigPolicy
import json
from otdd.otdd.pytorch.distance import DatasetDistance, FeatureCost
from utils.global_variable import DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH, SIGNIFICANCE_TRACE_PATH
from utils.data_loader import get_concat_dataset

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
import json
import argparse
import time
import random

import torch
from torchvision import models
import fcntl
import math

from utils.global_variable import SIGNIFICANCE_TRACE_PATH
        

class JobItem(object):
    def __init__(self, job_id, train_dataset_name, test_dataset_name, sub_test_key_id):
        self.job_id = job_id
        self.job_train_dataset_name = train_dataset_name
        self.job_test_dataset_name = test_dataset_name
        self.job_sub_test_key_id = sub_test_key_id

        self.history_acces = []
        self.history_delta_acces = []
        self.history_used_subtrain_block_per_epoches = []
        self.history_subtrain_block_used_count = {}
        
        self.history_significance_for_subtrain_datablock = {}

    def get_history_result(self, sub_train_key_id, history_rho):
        max_index = len(self.history_delta_acces) - 1
        history_result_fenzi = 0.0
        history_result_fenmu = 0.0
        # print("job {} first in {}".format(self.job_id, len(self.history_delta_acces)))
        for index, delta_acc in enumerate(self.history_delta_acces):
            add_delta_acc_value = 0.0
            if sub_train_key_id in self.history_used_subtrain_block_per_epoches[index]:
                add_delta_acc_value = delta_acc
            weight_rho = pow(history_rho, max_index - index)
            history_result_fenzi += weight_rho * add_delta_acc_value
            history_result_fenmu += weight_rho
        return history_result_fenzi / history_result_fenmu

class HISOTDDPolicy(SigPolicy):
    def __init__(self, logger, batch_size=16, history_alpha=0.5, history_rho=0.5):
        super().__init__()
        self._name = "HISOTDDPolicy"
        self.distance_batch_size = batch_size
        self.calculate_batch_size = batch_size
        self.OTDD_history_alpha = history_alpha
        self.history_rho = history_rho

        self.logger = logger

        self.need_update_backward = True

        self.OTDD_beta = 0.02
        self.max_OTDD = 0.0

        self.OTDD_trace_path = SIGNIFICANCE_TRACE_PATH + "/significance_OTDDPolicy.json"
        with open(self.OTDD_trace_path, "r+") as f:
            self.origin_OTDD_trace = json.load(f)
        
        self.current_OTDD = {}
        self.jobid_2_jobitem = {}

    def value_in_origin_OTDD_trace(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id):
        if train_dataset_name not in self.origin_OTDD_trace:
            return None
        if sub_train_key_id not in self.origin_OTDD_trace[train_dataset_name]:
            return None
        if test_dataset_name not in self.origin_OTDD_trace[train_dataset_name][sub_train_key_id]:
            return None
        if sub_test_key_id not in self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            return None
        return self.origin_OTDD_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id]

    def get_job_datablock_origin_OTDD_sync(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id):
        if train_dataset_name not in self.current_OTDD:
            self.current_OTDD[train_dataset_name] = {}
        if sub_train_key_id not in self.current_OTDD[train_dataset_name]:
            self.current_OTDD[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.current_OTDD[train_dataset_name][sub_train_key_id]:
            self.current_OTDD[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        if sub_test_key_id not in self.current_OTDD[train_dataset_name][sub_train_key_id][test_dataset_name]:
            result_d = self.value_in_origin_OTDD_trace(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
            self.current_OTDD[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = result_d    
            self.max_OTDD = max(self.max_OTDD, result_d)
        else:
            result_d = self.current_OTDD[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id]
        return result_d

    def get_job_significance_result_for_history_jobs_for_all_datablocks(self, history_job_id, all_significance_state):
        begin = time.time()
        origin_OTDDs = []
        norm_OTDDs = []
        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]

            # 获取原始的OTDD
            origin_otdd_d = self.get_job_datablock_origin_OTDD_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
            origin_OTDDs.append(origin_otdd_d)

        for origin_otdd in origin_OTDDs:
            norm_otdd = 1.0 / origin_otdd
            norm_OTDDs.append(norm_otdd)
        
        result = [
            norm_OTDDs[index] for index in range(len(all_significance_state))
        ]
        end = time.time()
        self.logger.info("history_job_id [{}] to datablocks significance: {}, time: {}".format(history_job_id, result, end-begin))
        return result

    def get_job_significance_result_for_all_datablocks(self, job_id, all_significance_state):
        begin = time.time()
        origin_OTDDs = []
        norm_OTDDs = []
        origin_history_accs = []
        norm_history_accs = []
        ucb_factors = []

        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]

            # 获取原始的OTDD
            origin_otdd_d = self.get_job_datablock_origin_OTDD_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
            origin_OTDDs.append(origin_otdd_d)

            # 获取原始的history_acc
            if job_id not in self.jobid_2_jobitem:
                self.jobid_2_jobitem[job_id] = JobItem(job_id, train_dataset_name, test_dataset_name, sub_test_key_id)
            
            if len(self.jobid_2_jobitem[job_id].history_acces) > 0:
                history_result = self.jobid_2_jobitem[job_id].get_history_result(sub_train_key_id, self.history_rho)
            else:
                history_result = 0.0
            origin_history_accs.append(history_result)

            # 获取UCB的factor
            if sub_train_key_id not in self.jobid_2_jobitem[job_id].history_subtrain_block_used_count:
                ucb_num = 0
            else:
                ucb_num = self.jobid_2_jobitem[job_id].history_subtrain_block_used_count[sub_train_key_id]
            current_times = len(self.jobid_2_jobitem[job_id].history_acces) + 1
            ucb_factor = math.sqrt(math.log(current_times) / (2 * (ucb_num + 1)))
            ucb_factors.append(ucb_factor)
        # 全局量
        for origin_otdd in origin_OTDDs:
            # norm_otdd = math.exp(-1.0 / (origin_otdd * self.OTDD_beta)) / math.exp(-1.0 / (self.max_OTDD * self.OTDD_beta))
            norm_otdd = 1.0 / origin_otdd # TODO(xlc): 因为在线场景中的区分度实在不高, 因此为了避免引入新的argue点, 还是选择了直接做除法
            norm_OTDDs.append(norm_otdd)
        # 局部量
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
        
            
        # 全局量 * (局部量 + UCB), 对量纲的影响是最小的(但是这样总会比历史记录更小啊...)
        result = [
            norm_OTDDs[index] * (norm_history_accs[index] + ucb_factors[index]) for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.info("job_id [{}] to datablocks significance: {}, time: {}".format(job_id, result, end-begin))
        return result

    '''
    def get_job_datablock_significance_sync(self, job_id, signficance_state, is_history):
        begin = time.time()
        train_dataset_name = signficance_state["train_dataset_name"]
        sub_train_key_id = signficance_state["sub_train_key_id"]
        test_dataset_name = signficance_state["test_dataset_name"]
        sub_test_key_id = signficance_state["sub_test_key_id"]
        
        if is_history:
            result_norm_d = self.get_job_datablock_origin_OTDD_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
        else:
            if job_id not in self.jobid_2_jobitem:
                self.jobid_2_jobitem[job_id] = JobItem(job_id, train_dataset_name, test_dataset_name, sub_test_key_id)
            if sub_train_key_id not in self.jobid_2_jobitem[job_id].history_significance_for_subtrain_datablock:
                self.jobid_2_jobitem[job_id].history_significance_for_subtrain_datablock[sub_train_key_id] = []
            
            if len(self.jobid_2_jobitem[job_id].history_significance_for_subtrain_datablock[sub_train_key_id]) > 0:
                history_significance = self.jobid_2_jobitem[job_id].history_significance_for_subtrain_datablock[sub_train_key_id][-1]
                if len(self.jobid_2_jobitem[job_id].history_acces) > 0:
                    history_result = self.jobid_2_jobitem[job_id].get_history_result(sub_train_key_id, self.history_rho)
                    result_norm_d = self.history_alpha * history_significance + (1 - self.history_alpha) * history_result
                else:
                    result_norm_d = self.history_alpha * history_significance
                self.jobid_2_jobitem[job_id].history_significance_for_subtrain_datablock[sub_train_key_id].append(result_norm_d)
            else:
                result_norm_d = self.get_job_datablock_origin_OTDD_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id)
                self.jobid_2_jobitem[job_id].history_significance_for_subtrain_datablock[sub_train_key_id].append(result_norm_d)
        end = time.time()
        self.logger.info("job_id [{}] to datablock [{}-{}] significance: {}".format(job_id, train_dataset_name, sub_train_key_id, result_norm_d))
        return result_norm_d
    '''
    
    def get_job_datablock_significance_async(self, job_id, signficance_state, device_index, is_history):
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
        train_loader = DataLoader(train_dataset, batch_size=self.distance_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.distance_batch_size)

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
                                batch_size=self.calculate_batch_size)

        
        result_d = dist.distance(maxsamples = 10000)
        print("result distance => [{}-{}-{}-{}]: {}".format(
            train_dataset_name, test_dataset_name, sub_train_key_id, sub_test_key_id, result_d
        ))
        self.current_OTDD[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = result_d
        self.max_OTDD = max(self.max_OTDD, result_d)

        result_norm_d = math.exp(-1.0 / (result_d * self.OTDD_beta)) / math.exp(-1.0 / (self.max_OTDD * self.OTDD_beta))

        if not is_history:
            if job_id not in self.jobid_2_jobitem:
                self.jobid_2_jobitem[job_id] = JobItem(job_id, test_dataset_name, sub_test_key_id)
            else:
                history_result = self.jobid_2_jobitem[job_id].get_history_result(sub_train_key_id, self.history_rho)
                result_norm_d = self.history_alpha * result_norm_d + (1 - self.history_alpha) * history_result
        self.logger.info("job_id [{}] to datablock [{}-{}] significance: {}".format(job_id, train_dataset_name, sub_train_key_id, result_norm_d))
        return result_norm_d
        
    def update_job_datablock_signficance_FAIR(self, job_id, used_sub_train_key_ids, current_result):
        # 需要有一些真实的结果来调整重要性评估指标, 将每一次计算得到的delta_loss反馈回场景中
        # 因为所有维度都固定, 所以delta_loss的变化幅度不会非常大?
        current_acc = current_result["test_acc"]
        if job_id not in self.jobid_2_jobitem:
            raise ValueError("job_id must in self.jobid_2_jobitem")
        job_item = self.jobid_2_jobitem[job_id]
        if len(job_item.history_acces) == 0:
            job_item.history_delta_acces.append(current_acc - 0.0)
        else:
            job_item.history_delta_acces.append(current_acc - job_item.history_acces[-1])
        job_item.history_acces.append(current_acc)
        job_item.history_used_subtrain_block_per_epoches.append(used_sub_train_key_ids)
        for sub_train_key in used_sub_train_key_ids:
            if sub_train_key not in job_item.history_subtrain_block_used_count:
                job_item.history_subtrain_block_used_count[sub_train_key] = 0
            job_item.history_subtrain_block_used_count[sub_train_key] += 1