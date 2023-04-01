import argparse
import json
import zerorpc
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision import models

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from utils.opacus_engine_tools import get_privacy_dataloader

from utils.global_variable import DATASET_PATH

import string

np.random.seed(42)
torch.manual_seed(42)

DATASET_SIZES = {
    'MNIST': (28,28),
    'FashionMNIST': (28,28),
    'EMNIST': (28,28),
    'QMNIST': (28,28),
    'KMNIST': (28,28),
    'USPS': (16,16),
    'SVHN': (32, 32),
    'CIFAR10': (32, 32),
    'STL10': (96, 96),
    'tiny-ImageNet': (64,64)
}

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307,), (0.3081,)),
    'USPS' : ((0.1307,), (0.3081,)),
    'FashionMNIST' : ((0.1307,), (0.3081,)),
    'QMNIST' : ((0.1307,), (0.3081,)),
    'EMNIST' : ((0.1307,), (0.3081,)),
    'KMNIST' : ((0.1307,), (0.3081,)),
    'ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'tiny-ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
}


def load_test_all_data(dataname, resize=28, to3channels=True, target='train', datadir=None):
    download = False
    if dataname in DATASET_NORMALIZATION.keys():
        transform_dataname = dataname
    else:
        transform_dataname = 'ImageNet'

    transform_list = []

    if dataname in ['MNIST', 'USPS', 'EMNIST'] and to3channels:
        transform_list.append(torchvision.transforms.Grayscale(3))

    transform_list.append(torchvision.transforms.ToTensor())
    transform_list.append(
        torchvision.transforms.Normalize(*DATASET_NORMALIZATION[transform_dataname])
    )

    if resize:
        if not dataname in DATASET_SIZES or DATASET_SIZES[dataname][0] != resize:
            ## Avoid adding an "identity" resizing
            transform_list.insert(0, transforms.Resize((resize, resize)))

    transform = transforms.Compose(transform_list)
    train_transform, valid_transform = transform, transform


    DATASET = getattr(torchvision.datasets, dataname)
    if dataname == 'EMNIST':
        split = 'bymerge'
        train = DATASET(datadir, split=split, train=True, download=download, transform=train_transform)
        test = DATASET(datadir, split=split, train=False, download=download, transform=valid_transform)
        ## EMNIST seems to have a bug - classes are wrong
        _merged_classes = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
        _all_classes = set(list(string.digits + string.ascii_letters))
        classes_split_dict = {
            'byclass': list(_all_classes),
            'bymerge': sorted(list(_all_classes - _merged_classes)),
            'balanced': sorted(list(_all_classes - _merged_classes)),
            'letters': list(string.ascii_lowercase),
            'digits': list(string.digits),
            'mnist': list(string.digits),
        }
        train.classes = classes_split_dict[split]
        if split == 'letters':
            ## The letters fold (and only that fold!!!) is 1-indexed
            train.targets -= 1
            test.targets -= 1
    elif dataname == 'STL10':
        train = DATASET(datadir, split='train', download=download, transform=train_transform)
        test = DATASET(datadir, split='test', download=download, transform=valid_transform)
        train.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        test.classes = train.classes
        train.targets = torch.tensor(train.labels)
        test.targets = torch.tensor(test.labels)
    elif dataname == 'SVHN':
        train = DATASET(datadir, split='train', download=download, transform=train_transform)
        test = DATASET(datadir, split='test', download=download, transform=valid_transform)
        ## In torchvision, SVHN 0s have label 0, not 10
        train.classes = test.classes = [str(i) for i in range(10)]
        train.targets = torch.tensor(train.labels)
        test.targets = torch.tensor(train.labels)
    elif dataname == 'LSUN':
        train = DATASET(datadir, classes='train', download=download, transform=train_transform)
    else:
        train = DATASET(datadir, train=True, download=download, transform=train_transform)
        test = DATASET(datadir, train=False, download=download, transform=valid_transform)

    if type(train.targets) is list:
        train.targets = torch.LongTensor(train.targets)
        test.targets  = torch.LongTensor(test.targets)

    if not hasattr(train, 'classes') or not train.classes:
        train.classes = sorted(torch.unique(train.targets).tolist())
        test.classes  = sorted(torch.unique(train.targets).tolist())

    if target == 'train':
        return train
    elif target == 'test':
        return test


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

def num_split(targets, sample_num):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    # 前10个
    num_used_num = int(sample_num / 10)
    n_classes = targets.max()+1
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(targets==y).flatten() 
           for y in range(n_classes)]
    result_idcs = []
    for index, ids in enumerate(class_idcs):
        if index < 10:
            print("check: {}".format(targets[ids[0]]))
            subset = np.random.choice(ids, num_used_num, replace=False).tolist()
            result_idcs.extend(subset)
    print(len(result_idcs))
    return result_idcs

if __name__ == "__main__":

    dataset_name = "EMNIST"
    raw_data_path = "/mnt/linuxidc_client/dataset/"
    result_subtest_config_path = "/mnt/linuxidc_client/dataset/{}/subtest.json".format(dataset_name)

    test_dataset_names = ['MNIST']
    test_sample_nums = [2000] # 从手写数字类中每个去200个

    current_subtest_config = {}
    with open(result_subtest_config_path, 'r+') as f:
        current_subtest_config = json.load(f)
        f.close()
    print(current_subtest_config)
    target_dataset_name = "{}-{}".format("_".join(test_dataset_names), "_".join([str(num) for num in test_sample_nums]))
    sub_test_key = "test_sub_0"
    current_subtest_config[target_dataset_name] = {}
    current_subtest_config[target_dataset_name][sub_test_key] = {}
    current_subtest_config[target_dataset_name][sub_test_key]["name"] = []
    current_subtest_config[target_dataset_name][sub_test_key]["path"] = raw_data_path
    current_subtest_config[target_dataset_name][sub_test_key]["indexes"] = []
    for i, name in enumerate(test_dataset_names):
        if test_sample_nums[i] > 0:
            test_dataset = load_test_all_data(name, resize=28, to3channels=True, target='test', datadir=raw_data_path)
            sample_indexes = num_split(test_dataset.targets, test_sample_nums[i])
            current_subtest_config[target_dataset_name][sub_test_key]["name"].append(name)
            # print(type(sample_indexes))
            current_subtest_config[target_dataset_name][sub_test_key]["indexes"].append(sample_indexes)
    
    # print(current_subtest_config)
    with open(result_subtest_config_path, 'w+') as f:
        json.dump(current_subtest_config, f)
        f.close()