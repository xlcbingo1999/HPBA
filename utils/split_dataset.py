import numpy as np
import random

import torch
from torch.utils.data import Subset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

import numpy as np
import matplotlib.pyplot as plt
import json


np.random.seed(42)
torch.manual_seed(42)

def dirichlet_split_noniid(train_labels, alpha, n_clients, same_capacity):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    if same_capacity:
        client_mean_label_num = int(len(train_labels) / n_clients)
        to_shared_index_list = []
        for id in range(len(client_idcs)):
            if len(client_idcs[id]) > client_mean_label_num:
                np.random.shuffle(client_idcs[id])
                to_shared_index_list.extend(client_idcs[id][client_mean_label_num:])
                client_idcs[id] = np.delete(client_idcs[id], range(client_mean_label_num, len(client_idcs[id])))
                # print("[after add] shared indexes num: {}".format(len(to_shared_index_list)))
        for id in range(len(client_idcs)):
            if len(client_idcs[id]) < client_mean_label_num:
                random.shuffle(to_shared_index_list)
                to_add_count = client_mean_label_num - len(client_idcs[id])
                client_idcs[id] = np.concatenate((client_idcs[id], to_shared_index_list[0:to_add_count]))
                to_shared_index_list = [ind for i, ind in enumerate(to_shared_index_list) if i >= to_add_count]
                # print("[after remove] shared indexes num: {}".format(len(to_shared_index_list)))
    return client_idcs

def split_and_draw(type, dataset, DIRICHLET_ALPHA, N_CLIENTS):
    input_sz, num_cls = dataset.data[0].shape[0],  len(dataset.classes)
    labels = np.array(dataset.targets)
    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = dirichlet_split_noniid(labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS, same_capacity=True)
    # 展示不同client的不同label的数据分布
    plt.figure(figsize=(20,3))
    plt.hist([labels[idc]for idc in client_idcs], stacked=True, 
            bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
            label=["{} Client {}".format(type, i) for i in range(N_CLIENTS)], rwidth=0.5)
    plt.xticks(np.arange(num_cls), dataset.classes)
    plt.legend()
    plt.savefig("/home/netlab/DL_lab/opacus_testbed/log_20230214/EMNIST_{}_dirichlet.png".format(type))
    return client_idcs