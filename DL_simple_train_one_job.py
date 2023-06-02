
from torchvision import models

import argparse
import json
import zerorpc
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision import models

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from utils.opacus_engine_tools import get_privacy_dataloader
from utils.global_variable import DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, TEST_DATASET_CONFIG_PATH
from utils.data_loader import get_concat_dataset
from utils.model_loader import PrivacyCNN, PrivacyFF
from tqdm import tqdm

import string
import os

def accuracy(preds, labels):
    return (preds == labels).mean()

parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=float, required=True)
parser.add_argument("--device_index", type=int, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--MAX_PHYSICAL_BATCH_SIZE", type=int, required=True)
args = parser.parse_args()
args_message = '\n'.join([f'{k}: {v}' for k, v in vars(args).items()])
print(args_message)

siton_run_epoch_num = 5
EPSILON_one_siton = args.epsilon
DELTA = 1e-8
MAX_GRAD_NORM = 1.2
train_datablock_num = 5
BATCH_SIZE = 1024
MAX_PHYSICAL_BATCH_SIZE = args.MAX_PHYSICAL_BATCH_SIZE
LR = 1e-3
train_dataset_name = "EMNIST"
sub_train_key_ids = [f"train_sub_{index}" for index in range(train_datablock_num)]
test_dataset_name = "EMNIST-2000"
sub_test_key_id = ["test_sub_0"]
device = torch.device(f'cuda:{args.device_index}')

train_dataset = get_concat_dataset(train_dataset_name, sub_train_key_ids, 
                                    DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, 
                                    "train")
print("finished load train_dataset")
test_dataset = get_concat_dataset(test_dataset_name, sub_test_key_id,
                                DATASET_PATH, TEST_DATASET_CONFIG_PATH,
                                "test")
print("finished load test_dataset")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

if args.model == "CNN":
    model = PrivacyCNN(output_dim=len(train_dataset.classes))
elif args.model == "FF":
    model = PrivacyFF(output_dim=len(train_dataset.classes))


if EPSILON_one_siton > 0.0:
    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=False)
    print("error: {}".format(errors))

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters


privacy_engine = PrivacyEngine() if EPSILON_one_siton > 0.0 else None
model, optimizer, train_loader = \
    get_privacy_dataloader(privacy_engine, model, optimizer, 
                            train_loader, siton_run_epoch_num, 
                            EPSILON_one_siton, DELTA, MAX_GRAD_NORM) 

# torch.cuda.reset_peak_memory_stats()

model.train()
for epoch in tqdm(range(siton_run_epoch_num)):
    total_train_loss = []
    total_train_acc = []
    if privacy_engine is not None:
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for i, (inputs, labels) in enumerate(memory_safe_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                output = model(inputs)
                
                loss = criterion(output, labels)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = labels.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    else:
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