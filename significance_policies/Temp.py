from significance_policies.BaseSigPolicy import SigPolicy
import json
import time

from utils.global_variable import DATASET_PATH, DATASET_CONFIG_NAME, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH, SIGNIFICANCE_TRACE_PREFIX_PATH
from utils.data_loader import get_concat_dataset
from utils.model_loader import PrivacyCNN, PrivacyFF

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import os
import numpy as np
from tqdm import tqdm
import multiprocessing

def accuracy(preds, labels):
    return (preds == labels).mean()

def cal_origin_Temp_loss(signficance_state, device_index, metric):
    
    print(f"begin: {signficance_state} in device_index {device_index} with metric {metric}")
    train_dataset_name = signficance_state["train_dataset_name"]
    test_dataset_name = signficance_state["test_dataset_name"]
    sub_train_key_id = signficance_state["sub_train_key_id"]
    sub_test_key_id = signficance_state["sub_test_key_id"]
    model_name = signficance_state["model_name"]
    model_config = signficance_state["model_config"]

    batch_size = model_config["batch_size"]
    LR = model_config["LR"]

    temp_loss_train_epoch_num = 30

    train_dataset = get_concat_dataset(train_dataset_name, sub_train_key_id, 
                            DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, 
                            "train")
    test_dataset = get_concat_dataset(test_dataset_name, sub_test_key_id,
                                    DATASET_PATH, TEST_DATASET_CONFIG_PATH,
                                    "test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
    device = torch.device("cuda:{}".format(device_index) if torch.cuda.is_available() else "cpu")
    if model_name == "CNN":
        model = PrivacyCNN(output_dim=len(train_dataset.classes))
    elif model_name == "FF":
        model = PrivacyFF(output_dim=len(train_dataset.classes))
    elif model_name == "resnet18":
        model = models.resnet18(num_classes=len(train_dataset.classes))

    model.train()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("model_name [{}] - epoch [{} to {}] begining ...".format(model_name, 0, temp_loss_train_epoch_num))
    for epoch in tqdm(range(temp_loss_train_epoch_num)):
        total_train_loss = []
        total_train_acc = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            total_train_loss.append(loss.item())

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            total_train_acc.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    final_val_loss = float("inf")
    final_val_acc = 0.0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        final_val_loss = loss.item()

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()
        acc = accuracy(preds, labels)
        final_val_acc = acc

    result = 0.0
    if metric == "Accuracy":
        result = float(final_val_acc)
    elif metric == "Loss":
        result = float(final_val_loss)
    print(f"====== finished result: {result} =======")
    return result

class TempPolicy(SigPolicy):
    def __init__(self, simulation, metric, logger):
        super().__init__()
        self._name = "TempPolicy"
        self.metric = metric
        self.need_update_backward = False
        self.max_temp_loss = 0.0

        if simulation:
            self.significance_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_TempPolicy_{DATASET_CONFIG_NAME}_{metric}.json"
        else:
            self.significance_trace_path = SIGNIFICANCE_TRACE_PREFIX_PATH + f"/significance_TempPolicy_{DATASET_CONFIG_NAME}_{metric}.json"
        self.logger = logger

        if os.path.exists(self.significance_trace_path):
            with open(self.significance_trace_path, "r+") as f:
                self.origin_significance_trace = json.load(f)
        else:
            self.origin_significance_trace = {}

    def set_origin_temp_loss_trace_value(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name, target_value):
        if train_dataset_name not in self.origin_significance_trace:
            self.origin_significance_trace[train_dataset_name] = {}
        if sub_train_key_id not in self.origin_significance_trace[train_dataset_name]:
            self.origin_significance_trace[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.origin_significance_trace[train_dataset_name][sub_train_key_id]:
            self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        if sub_test_key_id not in self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = {}
        self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id][model_name] = target_value

    def get_job_datablock_origin_temp_loss_sync(self, train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name):
        if train_dataset_name not in self.origin_significance_trace:
            self.origin_significance_trace[train_dataset_name] = {}
        if sub_train_key_id not in self.origin_significance_trace[train_dataset_name]:
            self.origin_significance_trace[train_dataset_name][sub_train_key_id] = {}
        if test_dataset_name not in self.origin_significance_trace[train_dataset_name][sub_train_key_id]:
            self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name] = {}
        if sub_test_key_id not in self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name]:
            self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id] = {}
        if model_name not in self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id]:
            device_index = 0
            device_index = 0
            signficance_state = {
                "train_dataset_name": train_dataset_name,
                "sub_train_key_id": sub_train_key_id,
                "test_dataset_name": test_dataset_name,
                "sub_test_key_id": sub_test_key_id,
                "model_name": model_name
            }
            result_d = cal_origin_Temp_loss(signficance_state, device_index)
            self.set_origin_temp_loss_trace_value(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name, result_d)
            self.max_OTDD = max(self.max_OTDD, result_d)
        else:
            result_d = self.origin_significance_trace[train_dataset_name][sub_train_key_id][test_dataset_name][sub_test_key_id][model_name]
        return result_d

    def write_to_origin_temp_loss_trace(self):
        self.logger.debug("==== write_to_origin_temp_loss_trace [origin_significance_trace] ====")
        self.logger.info(self.origin_significance_trace)
        with open(self.significance_trace_path, "w+") as f:
            json.dump(self.origin_significance_trace, f)

    def get_job_significance_result_for_all_datablocks(self, type_id, all_significance_state):
        begin = time.time()
        origin_Temps = []
        norm_Temps = []

        for index, signficance_state in enumerate(all_significance_state):
            train_dataset_name = signficance_state["train_dataset_name"]
            sub_train_key_id = signficance_state["sub_train_key_id"]
            test_dataset_name = signficance_state["test_dataset_name"]
            sub_test_key_id = signficance_state["sub_test_key_id"]
            model_name = signficance_state["model_name"]

            # 获取epsilon的剩余值
            # remain_epsilons.append(sub_train_key_remain_epsilon)

            # 获取原始的temp
            origin_temp_d = self.get_job_datablock_origin_temp_loss_sync(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name)
            origin_Temps.append(origin_temp_d)
            
        # 全局量
        for origin_temp in origin_Temps:
            if self.metric == "Accuracy":
                norm_temp = origin_temp
            elif self.metric == "Loss":
                norm_temp = 1.0 / origin_temp
            norm_Temps.append(norm_temp)
        
        # 全局量 * (局部量 + UCB), 对量纲的影响是最小的
        # 不能把当前时刻的remain_epsilon传进来, 会导致历史任务的价值偏高, 当前任务的价值不断下降
        # 太久没选的任务是否要将探索价值提高呢? 如果在世界时间中, 当最后的任务价值不断提高, 也会导致历史任务的价值不断提高...
        # 实际上很大概率就是任务在第一次被failed后, 整个系统会将其拒之门外...
        result = [
            norm_Temps[index] for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.debug("type_id [{}] to datablocks significance: {} [norm_Temps: {}], time: {}".format(
            type_id, result, norm_Temps, end-begin
        ))
        return result

    def get_job_datablock_significance_async(self, type_id, all_significance_state, cal_device_list):
        assert len(cal_device_list) > 0
        begin = time.time()

        origin_temp_losses = []
        norm_temp_losses = []
    
        group_size = len(cal_device_list)
        split_all_significance_state = [all_significance_state[i:i+group_size] for i in range(0, len(all_significance_state), group_size)]

        for sub_all_significance_state in tqdm(split_all_significance_state):
            with multiprocessing.Pool(processes=len(sub_all_significance_state)) as pool:
                metric = [self.metric] * len(sub_all_significance_state)
                args_zip = zip(sub_all_significance_state, cal_device_list, metric)
                origin_temp_loss_resultes = pool.starmap(cal_origin_Temp_loss, args_zip)

            for index, origin_temp_loss in enumerate(origin_temp_loss_resultes):
                signficance_state = sub_all_significance_state[index]
                train_dataset_name = signficance_state["train_dataset_name"]
                test_dataset_name = signficance_state["test_dataset_name"]
                sub_train_key_id = signficance_state["sub_train_key_id"]
                sub_test_key_id = signficance_state["sub_test_key_id"]
                model_name = signficance_state["model_name"]
                
                self.set_origin_temp_loss_trace_value(train_dataset_name, sub_train_key_id, test_dataset_name, sub_test_key_id, model_name, origin_temp_loss)    
                self.max_temp_loss = max(self.max_temp_loss, origin_temp_loss)
                origin_temp_losses.append(origin_temp_loss)
                self.logger.debug("result distance => [{}-{}-{}-{}]: {}".format(
                    train_dataset_name, test_dataset_name, sub_train_key_id, sub_test_key_id, origin_temp_loss
                ))
            
            self.write_to_origin_temp_loss_trace()

        for origin_temp_loss in origin_temp_losses:
            # norm_otdd = math.exp(-1.0 / (origin_otdd * self.OTDD_beta)) / math.exp(-1.0 / (self.max_temp_loss * self.OTDD_beta))
            norm_temp_loss = 1.0 / origin_temp_loss # TODO(xlc): 因为在线场景中的区分度实在不高, 因此为了避免引入新的argue点, 还是选择了直接做除法
            norm_temp_losses.append(norm_temp_loss)

        result = [
            norm_temp_losses[index] for index in range(len(all_significance_state))
        ]
        
        end = time.time()
        self.logger.debug("type_id [{}] to datablocks significance: {} [norm_temp_losses: {}], time: {}".format(
            type_id, result, norm_temp_losses, end-begin
        ))
        return result