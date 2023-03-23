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

from utils.data_loader import load_torchvision_data_from_indexes

class OTDDPolicy(SigPolicy):
    def __init__(self, batch_size):
        super().__init__()
        self._name = "OTDDPolicy"
        self.distance_batch_size = batch_size
        self.calculate_batch_size = batch_size

        # self.significance_trace = {}
        # self.significance_trace_path = SIGNIFICANCE_TRACE_PATH + "/significance_{}.json".format(self.name)
        # with open(self.significance_trace_path, "r+") as f:
        #     fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        #     self.significance_trace = json.load(f)
        #     fcntl.flock(f, fcntl.LOCK_UN)

    def get_job_datablock_significance_sync(self, signficance_state):
        begin = time.time()
        train_dataset_name = signficance_state["train_dataset_name"]
        sub_train_key_ids = signficance_state["sub_train_key_ids"]
        hashed_sub_train_key_ids = "_".join(sub_train_key_ids)
        test_dataset_name = signficance_state["test_dataset_name"]
        sub_test_key_ids = signficance_state["sub_test_key_ids"]
        hashed_sub_test_key_ids = "_".join(sub_test_key_ids)
        if train_dataset_name not in self.significance_trace:
            self.significance_trace[train_dataset_name] = {}
        if test_dataset_name not in self.significance_trace[train_dataset_name]:
            self.significance_trace[train_dataset_name][test_dataset_name] = {}
        if hashed_sub_train_key_ids not in self.significance_trace[train_dataset_name][test_dataset_name]:
            self.significance_trace[train_dataset_name][test_dataset_name][hashed_sub_train_key_ids] = {}
        if hashed_sub_test_key_ids not in self.significance_trace[train_dataset_name][test_dataset_name][hashed_sub_train_key_ids]:
            self.significance_trace[train_dataset_name][test_dataset_name][hashed_sub_train_key_ids][hashed_sub_test_key_ids] = {}
            result_d = None
        else:
            result_d = self.significance_trace[train_dataset_name][test_dataset_name][hashed_sub_train_key_ids][hashed_sub_test_key_ids]
        end = time.time()
        print("calculate OTDD time {}".format(end - begin))
        return result_d

    def get_job_datablock_significance_async(self, signficance_state, device_index):
        train_dataset_name = signficance_state["train_dataset_name"]
        test_dataset_name = signficance_state["test_dataset_name"]
        sub_train_key_ids = signficance_state["sub_train_key_ids"]
        hashed_sub_train_key_ids = "_".join(sub_train_key_ids)
        sub_test_key_ids = signficance_state["sub_test_key_ids"]
        hashed_sub_test_key_ids = "_".join(sub_test_key_ids)
        # 耗时操作, 计算OTDD
        train_dataset = get_concat_dataset(train_dataset_name, sub_train_key_ids, 
                                DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, 
                                "train")
        test_dataset = get_concat_dataset(test_dataset_name, sub_test_key_ids,
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
            train_dataset_name, test_dataset_name, hashed_sub_train_key_ids, hashed_sub_test_key_ids, result_d
        ))
        return result_d

    def update_job_datablock_signficance(self, signficance_state):
        print("TODO!!!!")