import numpy as np
import random

import torch
from torch.utils.data import SubsetRandomSampler, Dataset
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
    if n_clients == 1:
        client_idcs = [[] for _ in range(n_clients)]
        client_idcs[0] = np.array(range(len(train_labels)))
        return client_idcs
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

def split_and_draw(type, dataset_name, dataset, DIRICHLET_ALPHA, N_CLIENTS):
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
    plt.savefig("/mnt/linuxidc_client/dataset/EMNIST/{}_{}_dirichlet_{}_{}.png".format(dataset_name, type, DIRICHLET_ALPHA, N_CLIENTS))
    return client_idcs

if __name__ == "__main__":
    TRAIN_N_CLIENTS = 144
    TRAIN_DIRICHLET_ALPHA = 1.0

    dataset_name = "EMNIST"
    raw_data_path = "/mnt/linuxidc_client/dataset/"
    result_subtrain_config_path = "/mnt/linuxidc_client/dataset/{}/subtrain_{}_split_{}_dirichlet.json".format(dataset_name, TRAIN_N_CLIENTS, TRAIN_DIRICHLET_ALPHA)
    label_type = "bymerge"
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = EMNIST(
        root=raw_data_path,
        split=label_type,
        download=False,
        train=True,
        transform=transform
    )

    all_train_distribution = train_dataset.targets.unique(return_counts=True)
    # all_test_distribution = test_dataset.targets.unique(return_counts=True)
    print("train_dataset: {}".format(len(train_dataset)))
    # print("test_dataset: {}".format(len(test_dataset)))
    print("all_train_distribution: {}".format(all_train_distribution))
    # print("all_test_distribution: {}".format(all_test_distribution))

    train_client_idcs = split_and_draw("train", dataset_name, train_dataset, TRAIN_DIRICHLET_ALPHA, TRAIN_N_CLIENTS)

    current_subtrain_config = {}
    current_subtrain_config[dataset_name] = {}
    for index, client_indexes in enumerate(train_client_idcs):
        sub_train_key = "train_sub_{}".format(index)
        sub_train_dataset = CustomDataset(train_dataset, client_indexes)
        sub_train_distribution = {
            str(target.item()): 0 for target in all_train_distribution[0]
        }
        temp_distribution = sub_train_dataset.get_class_distribution()
        for index in range(len(temp_distribution[0])):
            target = temp_distribution[0][index]
            num = temp_distribution[1][index]
            sub_train_distribution[str(target.item())] = num.item()
        print("sub_train_distribution: ", sub_train_distribution)
        
        current_subtrain_config[dataset_name][sub_train_key] = {}
        # current_subtrain_config[dataset_name][sub_train_key]["label_type"] = label_type
        # current_subtrain_config[dataset_name][sub_train_key]["label_distribution"] = sub_train_distribution
        current_subtrain_config[dataset_name][sub_train_key]["name"] = [dataset_name]
        current_subtrain_config[dataset_name][sub_train_key]["path"] = raw_data_path
        current_subtrain_config[dataset_name][sub_train_key]["indexes"] = [client_indexes.tolist()]
        current_subtrain_config[dataset_name][sub_train_key]["length"] = [len(client_indexes)]
        
    with open(result_subtrain_config_path, 'w+') as f:
        json.dump(current_subtrain_config, f)
        f.close()