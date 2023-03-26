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
        self.history_used_subtrain_block = []
        self.current_norm_OTDD = {}
        self.current_norm_loss_sig = {}
        self.current_significance = {}

    def get_history_result(self, sub_train_key_id, history_rho):
        max_index = len(self.history_delta_acces) - 1
        history_result_fenzi = 0.0
        history_result_fenmu = 0.0
        # print("job {} first in {}".format(self.job_id, len(self.history_delta_acces)))
        for index, delta_acc in enumerate(self.history_delta_acces):
            add_delta_acc_value = 0.0
            if sub_train_key_id in self.history_used_subtrain_block[index]:
                add_delta_acc_value = delta_acc
            weight_rho = pow(history_rho, max_index - index)
            history_result_fenzi += weight_rho * add_delta_acc_value
            history_result_fenmu += weight_rho
        return history_result_fenzi / history_result_fenmu

class HISOTDDPolicy(SigPolicy):
    def __init__(self, logger, batch_size=16, history_alpha=0.2, history_rho=0.5):
        super().__init__()
        self._name = "HISOTDDPolicy"
        self.distance_batch_size = batch_size
        self.calculate_batch_size = batch_size
        self.history_alpha = history_alpha
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

    def get_job_datablock_significance_sync(self, job_id, signficance_state, is_history):
        begin = time.time()
        train_dataset_name = signficance_state["train_dataset_name"]
        sub_train_key_id = signficance_state["sub_train_key_id"]
        test_dataset_name = signficance_state["test_dataset_name"]
        sub_test_key_id = signficance_state["sub_test_key_id"]
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
        
        result_norm_d = math.exp(-1.0 / (result_d * self.OTDD_beta)) / math.exp(-1.0 / (self.max_OTDD * self.OTDD_beta))

        if not is_history:
            if job_id not in self.jobid_2_jobitem:
                self.jobid_2_jobitem[job_id] = JobItem(job_id, train_dataset_name, test_dataset_name, sub_test_key_id)
            if len(self.jobid_2_jobitem[job_id].history_acces) > 0:
                history_result = self.jobid_2_jobitem[job_id].get_history_result(sub_train_key_id, self.history_rho)
                result_norm_d = self.history_alpha * result_norm_d + (1 - self.history_alpha) * history_result
        end = time.time()
        self.logger.info("job_id [{}] to datablock [{}-{}] significance: {}".format(job_id, train_dataset_name, sub_train_key_id, result_norm_d))
        return result_norm_d

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
        job_item.history_used_subtrain_block.append(used_sub_train_key_ids)