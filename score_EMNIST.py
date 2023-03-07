import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

import json
import time

from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        return x, y
    
    def get_class_distribution(self):
        sub_targets = self.targets[self.indices]
        return sub_targets.unique(return_counts=True)

dataset_name = 'EMNIST'
raw_data_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST'
sub_train_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/sub_train_datasets_config.json'
sub_test_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/test_dataset_config.json'

with open(sub_train_config_path, 'r+') as f:
    current_subtrain_config = json.load(f)
    f.close()
with open(sub_test_config_path, 'r+') as f:
    current_subtest_config = json.load(f)
    f.close()

train_keys = current_subtrain_config[dataset_name].keys()
test_keys = current_subtest_config[dataset_name].keys()
train_keys = ["train_sub_0"]
test_keys = ["test_sub_0"]

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

train_dataset = EMNIST(
    root=raw_data_path,
    split="bymerge",
    download=False,
    train=True,
    transform=transform
)
test_dataset = EMNIST(
    root=raw_data_path,
    split="bymerge",
    download=False,
    train=False,
    transform=transform
)

def scale_normal(datasets, datasets_test):
    """
        Scale both the training and test set to standard normal distribution. The training set is used to fit.
        Args:
            datasets (list): list of datasets of length M
            datasets_test (list): list of test datasets of length M
        
        Returns:
            two lists containing the standarized training and test dataset
    """
    scaler = StandardScaler()
    scaler.fit(torch.vstack(datasets))
    return [torch.from_numpy(scaler.transform(dataset)).float() for dataset in datasets], [torch.from_numpy(scaler.transform(dataset)).float() for dataset in datasets_test]


def compute_volumes(datasets):
    d = datasets[0].shape[1]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reshape(-1 ,d)

    X = np.concatenate(datasets, axis=0).reshape(-1, d)
    volumes = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        volumes[i] = np.sqrt(np.linalg.det( dataset.T @ dataset ) + 1e-8)

    volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8).round(3)
    return volumes, volume_all

for sub_train_key in train_keys:
    for sub_test_key in test_keys:
        print("check result in train_key: {} and test_key: {}".format(sub_train_key, sub_test_key))
        real_train_index = current_subtrain_config[dataset_name][sub_train_key]["indexes"]
        real_test_index = current_subtest_config[dataset_name][sub_test_key]["indexes"]

        sub_train_dataset = CustomDataset(train_dataset, real_train_index)
        sub_test_dataset = CustomDataset(test_dataset, real_test_index)
        feature_train = tuple([sub_train_dataset.dataset.data[real_train_index].view(sub_train_dataset.dataset.data[real_train_index].size(0), -1) ]) 
        feature_test = tuple([sub_test_dataset.dataset.data[real_test_index].view(sub_test_dataset.dataset.data[real_test_index].size(0), -1) ])

        print("Finished split datasets!")
        feature_train_result, feature_test_result = scale_normal(feature_train, feature_test)
        print("feature_train_result: ", feature_train_result[0].sum())
        print("feature_test_result: ", feature_test_result[0].sum())
        volumes, vol_all = compute_volumes(feature_train_result)

        print("volumes: ", volumes)
        print("vol_all: ", vol_all)

        


