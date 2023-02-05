from cmath import nan
from collections import Counter
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import os
import random
from pathlib import Path

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import nltk
import re
from utils.global_variable import GLOBAL_PATH, DATASET_PATH
import pickle
from tqdm import tqdm

class CustomVisionDataset(Dataset):
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

    def get_subset_targets(self):
        return [self.targets[idx] for idx in self.indices]
    
class CustomTextDataset(TensorDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.targets = [int(tar) for tar in dataset.tensors[1]] # 保留targets属性


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        return x, y

    def get_subset_targets(self):
        return [self.targets[idx] for idx in self.indices]

def dirichlet_split_noniid(train_all_dataset, alpha, n_clients, same_capacity, plot_path):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    train_labels = np.array(train_all_dataset.get_subset_targets())
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
        if len(to_shared_index_list) > 0:
            choice_client_id = random.choice(range(len(client_idcs)))
            client_idcs[choice_client_id] = np.concatenate((client_idcs[choice_client_id], to_shared_index_list))
            to_shared_index_list.clear()

        assert len(to_shared_index_list) == 0

        label_check = [0] * len(train_labels)
        for c in client_idcs:
            for id in c:
                label_check[id] += 1
                if label_check[id] > 1:
                    raise ValueError("sdsd")
        for c in label_check:
            if c <= 0:
                raise ValueError("sdsd2")
    # print(labels_target)
    result_idcs = [[] for _ in range(n_clients)]
    for i, idc in enumerate(client_idcs): # 一个向量
        temp_idcs = [train_all_dataset.indices[si] for si in idc]
        result_idcs[i] = temp_idcs
    if plot_path is not None:
        plt.figure(figsize=(20,3))
        plt.hist([train_labels[idc] for idc in client_idcs], stacked=True, 
                bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
                label=["Client {}".format(i) for i in range(n_clients)], rwidth=0.5)
        plt.legend()
        plt.savefig(plot_path)
        print("save result success!")
    return result_idcs

def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

def dispatch_indexes_to_train_valid(all_dataset, label_num, num_valid):
    assert len(all_dataset) > num_valid
    sample_num_per_label = int(num_valid / label_num)
    valid_idx_per_label = [[] for _ in range(label_num)]
    train_idxs = []
    for index, label in enumerate(all_dataset.targets):
        if len(valid_idx_per_label[label]) < sample_num_per_label:
            valid_idx_per_label[label].append(index)
        else:
            train_idxs.append(index)
    flatten_valid_idxs = flatten(valid_idx_per_label)
    return train_idxs, flatten_valid_idxs

    

def get_dataset_multi_split_criteo(name, VALID_SIZE):
    dataset_root_map = {
        'Beauty': DATASET_PATH + '/criteo_dataset/dac/',
    }
    DATA_ROOT = dataset_root_map[name]

    x_train, x_val, y_train, y_val = None, None, None, None
    # do sth
    continue_var = ['I' + str(i) for i in range(1, 14)]
    cat_features = ['C' + str(i) for i in range(1, 27)]
    
    col_names_train = ['Label'] + continue_var + cat_features

    train_path = DATA_ROOT + 'sub_train.csv'
    
    start = time.time()
    train = pd.read_csv(train_path)
    print('Reading data costs %.2f seconds'%(time.time() - start))

    print('train has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    ls = list(train.columns)
    ls.remove('Label')
    for col in continue_var:
        print("dtype: {}".format(train[col].dtype))
        print('filling NA value of {} ...'.format(col))

        empty_string_mask = train[col].isnull().sum() # True和False
        print("stringify check nan count: {}".format(empty_string_mask))
        train[col] = train[col].fillna(train[col].mean())
        # train[col] = train[col].astype('float64')

    train = train.fillna('unknown')

    y_train = train[['Label']]
    x_train = train.drop(['Label'], axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VALID_SIZE, stratify=y_train, random_state=256)

    print("x_train: ", x_train)
    print("y_train: ", y_train)

    print("x_val: ", x_val)
    print("y_val: ", y_val)
    
    return x_train, x_val, y_train, y_val   

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, review in enumerate(sentences):
        if len(review) != 0:
            features[i, -len(review):] = np.array(review)[:seq_len]
    return features

def get_split_train_dataset_index(combine_years_months_days, SPLIT_NUM, same_capacity):
    assert same_capacity == True
    split_train_dataset_indexes_result = []
    if same_capacity:
        len_train_dataset = len(combine_years_months_days)
        all_index = list(range(len_train_dataset))
        batch_size = int(len_train_dataset / SPLIT_NUM)
        split_train_dataset_indexes_result = [
            all_index[i: i+batch_size] for i in range(0, len_train_dataset, batch_size)
        ]
        split_train_dataset_indexes_result[-2].extend(split_train_dataset_indexes_result[-1])
        split_train_dataset_indexes_result.pop(-1)
    return split_train_dataset_indexes_result

def extract_data(content_review): #Returns: (sentence,label)
    sentiment = list(content_review['sentiment'])
    overall = list(content_review['overall'])
    sentences = list(content_review['reviews'])
    years = list(content_review['year'])
    months = list(content_review['month'])
    days = list(content_review['day'])
    return sentences, sentiment, overall, years, months, days

def get_review_dataset_combine_split(categorys, label_type, GROUP_NUM, VALID_SIZE, SEQUENCE_LENGTH, BATCH_SIZE, SPLIT_NUM, ALPHA, same_capacity):
    MIX_EUQA_PREFIX = DATASET_PATH + '/Amazon_Review/reviews_MIX3_{}_5_'.format(label_type)
    TOKENIZE_PATH = DATASET_PATH + '/Amazon_Review/tokenize/reviews_MIX3_{}_5_{}_'.format(label_type, VALID_SIZE)
    all_contents_review = []
    for category in tqdm(categorys):
        EXTENSION_CSV_PATH_PREFIX = DATASET_PATH + '/Amazon_Review/reviews_{}_5_extension.csv'.format(category)
        category_content_review = pd.read_csv(EXTENSION_CSV_PATH_PREFIX)
        all_contents_review.append(category_content_review)
    all_contents_review = pd.concat(all_contents_review)
    print(all_contents_review.info())
    # calling the label encoder function
    label_encoder = preprocessing.LabelEncoder() 
    all_contents_review['sentiment']= label_encoder.fit_transform(all_contents_review['sentiment']) 
    print(all_contents_review['sentiment'].unique())
    print(all_contents_review['sentiment'].value_counts())
    print(all_contents_review['overall'].value_counts())

    # 这里要根据Sentiment和overall的数量进行一波采样
    assert label_type == 'sentiment' or label_type == 'overall'
    group_obj = all_contents_review.groupby(label_type)
    train_contents_review = []
    valid_contents_review = []
    for group_idx, obj in group_obj:
        assert len(obj) >= GROUP_NUM
        sample_train_obj = obj.sample(GROUP_NUM)
        sample_train_obj = sample_train_obj.dropna()
        # print("sample_obj: {}".format(sample_obj))
        remain_obj = obj.drop(sample_train_obj['Unnamed: 0.1'])
        max_valid_num = int(GROUP_NUM * VALID_SIZE)
        if len(remain_obj) < max_valid_num:
            max_valid_num = len(remain_obj)
        sample_valid_obj = remain_obj.sample(max_valid_num)
        sample_valid_obj = sample_valid_obj.dropna()
        train_contents_review.append(sample_train_obj)
        valid_contents_review.append(sample_valid_obj)
    train_contents_review = pd.concat(train_contents_review)
    valid_contents_review = pd.concat(valid_contents_review)
    
    train_contents_review = train_contents_review.reset_index(drop=True)
    valid_contents_review = valid_contents_review.reset_index(drop=True)
    print(train_contents_review.info())
    print(valid_contents_review.info())
    train_contents_review.to_csv(MIX_EUQA_PREFIX + "train_extension.csv")
    valid_contents_review.to_csv(MIX_EUQA_PREFIX + "valid_extension.csv")

    nltk.download('punkt') # Tokenizer
    words = Counter()

    train_sentences, train_sentiment, train_overall, train_years, train_months, train_days = extract_data(train_contents_review)
    valid_sentences, valid_sentiment, valid_overall, valid_years, valid_months, valid_days  = extract_data(valid_contents_review)

    train_sentences = list(filter(None, train_sentences))
    valid_sentences = list(filter(None, valid_sentences))
    train_sentences = [sen for sen in train_sentences if sen is not np.nan]
    valid_sentences = [sen for sen in valid_sentences if sen is not np.nan]
    print(f'Valid Training Sentences: {len(train_sentences)}')
    print(f'Valid valid Sentences: {len(valid_sentences)}')

    for i, sentence in enumerate(tqdm(train_sentences)):
        try:
            #tokens = nltk.word_tokenize(sentence)
            tokens = nltk.regexp_tokenize(sentence, pattern="\s+", gaps = True)
            train_sentences[i] = []
            for word in tokens: # Tokenize the words
                words.update([word.lower()]) # To Lower Case
                train_sentences[i].append(word)
        except:
            print(sentence)
    print(f"tokenize 100% done")

    # Remove infrequent words (i.e. words that only appear once)
    words = {k:v for k,v in words.items() if v>1}
    # Sort the words according to frequency, descending order
    words = sorted(words, key=words.get, reverse=True)
    # Add padding & unknown to corpus
    words = ['_PAD','_UNK'] + words

    # Dictionaries for fast mappings
    word2idx = {w:i for i,w in enumerate(words)}
    idx2word = {i:w for i,w in enumerate(words)}

    # Convert word to indices
    for i, sentence in enumerate(tqdm(train_sentences)):
        train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
        
    for i, sentence in enumerate(tqdm(valid_sentences)):
        # For valid sentences, we have to tokenize the sentences as well
        valid_sentences[i] = [word2idx[word.lower()] if word in word2idx else 0 for word in nltk.regexp_tokenize(sentence, pattern="\s+", gaps = True)]
    print(f"word2idx 100% done")

    print("save begin:")
    with open(TOKENIZE_PATH + 'train_sentences.data', 'wb') as file:
        pickle.dump(train_sentences, file)
    with open(TOKENIZE_PATH + 'valid_sentences.data', 'wb') as file:
        pickle.dump(valid_sentences, file)
    with open(TOKENIZE_PATH + 'train_sentiment.data', 'wb') as file:
        pickle.dump(train_sentiment, file)
    with open(TOKENIZE_PATH + 'valid_sentiment.data', 'wb') as file:
        pickle.dump(valid_sentiment, file)
    with open(TOKENIZE_PATH + 'train_overall.data', 'wb') as file:
        pickle.dump(train_overall, file)
    with open(TOKENIZE_PATH + 'valid_overall.data', 'wb') as file:
        pickle.dump(valid_overall, file)
    with open(TOKENIZE_PATH + 'train_years.data', 'wb') as file:
        pickle.dump(train_years, file)
    with open(TOKENIZE_PATH + 'valid_years.data', 'wb') as file:
        pickle.dump(valid_years, file)
    with open(TOKENIZE_PATH + 'train_months.data', 'wb') as file:
        pickle.dump(train_months, file)
    with open(TOKENIZE_PATH + 'valid_months.data', 'wb') as file:
        pickle.dump(valid_months, file)
    with open(TOKENIZE_PATH + 'train_days.data', 'wb') as file:
        pickle.dump(train_days, file)
    with open(TOKENIZE_PATH + 'valid_days.data', 'wb') as file:
        pickle.dump(valid_days, file)
    with open(TOKENIZE_PATH + 'words.data', 'wb') as file:
        pickle.dump(words, file)

def get_review_dataset_multi_split(category, label_type, VALID_SIZE, SEQUENCE_LENGTH, SPLIT_NUM, same_capacity):
    load_types = [
        'train_sentences', 'valid_sentences',
        'train_sentiment', 'valid_sentiment',
        'train_overall', 'valid_overall',
        'train_years', 'valid_years',
        'train_months', 'valid_months',
        'train_days', 'valid_days',
        'words'
    ]
    all_datas = {}
    TOKEN_PATH_PREFIX = DATASET_PATH + '/Amazon_Review/tokenize/reviews_{}_5_{}_'.format(category, VALID_SIZE)

    for type in load_types:
        with open(TOKEN_PATH_PREFIX + '{}.data'.format(type), 'rb') as file:
            new_file_list = pickle.load(file)
            all_datas[type] = new_file_list
        if type == 'valid_overall' or type == 'train_overall':
            all_datas[type] = [overall - 1 for overall in all_datas[type]]
        print("len({}): {}".format(type, len(all_datas[type])))
    all_data_len = len(all_datas['train_years'])
    combine_years_months_days = [
        list((all_datas['train_years'][i],
        all_datas['train_months'][i],
        all_datas['train_days'][i]))
    for i in range(all_data_len)]
    
    label_type_count = Counter(all_datas['train_{}'.format(label_type)])
    print(label_type_count)
    # combine_dtype = [('year', int), ('month', int), ('day', int)]
    combine_years_months_days = pd.DataFrame(combine_years_months_days, columns=['year', 'month', 'day'])
    sort_combine_years_months_days = combine_years_months_days.sort_values(by=['year', 'month', 'day']).reset_index()
    print(sort_combine_years_months_days)

    train_filter_index = list(sort_combine_years_months_days['index']) # 通过这个list去控制filter
    print("max_index: {}; min_index: {}".format(max(train_filter_index), min(train_filter_index)))

    for type in load_types:
        if type[0:6] == 'train_':
            all_datas[type] = [all_datas[type][nidx] for nidx in train_filter_index]
            print('{} reindex train finished!'.format(type))

    combine_years_months_days = [list((all_datas['train_years'][i], all_datas['train_months'][i], all_datas['train_days'][i]))
                                    for i in range(all_data_len)]
    all_meta_sentences = torch.from_numpy(pad_input(all_datas['train_sentences'], SEQUENCE_LENGTH))
    valid_sentences = torch.from_numpy(pad_input(all_datas['valid_sentences'], SEQUENCE_LENGTH))
    if label_type == 'sentiment':
        label_num = len(set(all_datas['train_sentiment']) | set(all_datas['valid_sentiment']))
        all_meta_sentiment = torch.from_numpy(np.array(all_datas['train_sentiment']))
        valid_sentiment = torch.from_numpy(np.array(all_datas['valid_sentiment']))
        all_train_dataset = TensorDataset(
            all_meta_sentences,
            all_meta_sentiment
        )
        valid_data = TensorDataset(
            valid_sentences,
            valid_sentiment
        )
    elif label_type == 'overall':
        label_num = len(set(all_datas['train_overall']) | set(all_datas['valid_overall']))
        all_meta_overall = torch.from_numpy(np.array(all_datas['train_overall']))
        valid_overall_np = np.array(all_datas['valid_overall'])
        valid_overall = torch.from_numpy(np.array(all_datas['valid_overall']))
        all_train_dataset = TensorDataset(
            all_meta_sentences,
            all_meta_overall
        )
        
        valid_data = TensorDataset(
            valid_sentences,
            valid_overall
        )
    else:
        raise ValueError("No this label type!")

    print("load valid dataset finished! len: {}".format(len(valid_data)))
    print("load all dataset finished! len: {}".format(len(all_train_dataset)))

    split_train_dataset_index = get_split_train_dataset_index(combine_years_months_days, SPLIT_NUM, same_capacity)
    train_sentences_matrix = []
    train_data_list = []
    
    for index_list in split_train_dataset_index:
        current_train_dataset = CustomTextDataset(all_train_dataset, index_list)
        train_data_list.append(current_train_dataset)
        print("load sub dataset finished! len: {}".format(len(current_train_dataset)))

    word2idx = {w:i for i,w in enumerate(all_datas['words'])}
    # others = {
    #     'vocab_size': len(word2idx) + 1
    # }
    vocab_size = len(word2idx) + 1

    len_all_train = len(all_train_dataset)
    all_train_dataset = CustomTextDataset(all_train_dataset, range(len_all_train))
    len_all_valid = len(valid_data)
    valid_data = CustomTextDataset(valid_data, range(len_all_valid))
    return all_train_dataset, train_data_list, valid_data, label_num, vocab_size

def get_image_dataset_mutli_split(name, VALID_SIZE, BATCH_SIZE, SPLIT_NUM, ALPHA, same_capacity, plot_path):
    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    dataset_root_map = {
        'CIFAR10': DATASET_PATH + '/cifar10',
        'CIFAR100': DATASET_PATH + '/cifar10',
        'MiniImageNet': DATASET_PATH + '/imagenet/data/',
        'CINIC10': DATASET_PATH + '/cinic/',
    }
    dataset_mean_map = {
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'MiniImageNet': (0.485, 0.456, 0.406),
        'CINIC10': (0.47889522, 0.47227842, 0.43047404),
    }
    dataset_std_map = {
        'CIFAR10': (0.2023, 0.1994, 0.2010),
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'MiniImageNet': (0.229, 0.224, 0.225),
        'CINIC10': (0.24205776, 0.23828046, 0.25874835),
    }
    has_origin_valid_dataset = {
        'CIFAR10': False,
        'CIFAR100': False,
        'MiniImageNet': False,
        'CINIC10': True,
    }
    DATA_ROOT = dataset_root_map[name]
    MEAN = dataset_mean_map[name]
    STD_DEV = dataset_std_map[name]
    
    all_dataset = None
    train_all_dataset = None
    valid_all_dataset = None
    if name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD_DEV),
        ])
        all_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    elif name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD_DEV),
        ])
        all_dataset = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform)
    elif name == 'MiniImageNet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            # transforms.RandomCrop(84, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD_DEV),
        ])
        all_dataset = ImageFolder(DATA_ROOT, transform=transform)
    elif name == 'CINIC10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            # transforms.RandomCrop(84, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD_DEV),
        ])
        train_all_dataset = ImageFolder(DATA_ROOT + 'train/', transform=transform)
        valid_all_dataset = ImageFolder(DATA_ROOT + 'valid/', transform=transform)
    else:
        raise ValueError("CIFAR has not this class for using!")
    if has_origin_valid_dataset[name]:
        assert (train_all_dataset is not None and valid_all_dataset is not None and all_dataset is None)
    else:
        assert (train_all_dataset is None and valid_all_dataset is None and all_dataset is not None)

    if not has_origin_valid_dataset[name]:
        num_all = len(all_dataset)
        label_num = len(set(all_dataset.classes))
        indices = list(range(num_all))
        num_train = int(np.floor((1 - VALID_SIZE) * num_all))
        num_valid = int(np.floor(VALID_SIZE * num_all))
        print("label_num: {}".format(label_num))

        train_idx, valid_idx = dispatch_indexes_to_train_valid(all_dataset, label_num, num_valid)
        train_all_dataset = CustomVisionDataset(all_dataset, train_idx)
        valid_all_dataset = CustomVisionDataset(all_dataset, valid_idx)
    else:
        label_num = len(set(train_all_dataset.classes) | set(valid_all_dataset.classes))
        train_idxs = range(len(train_all_dataset))
        valid_idxs = range(len(valid_all_dataset))
        print("train_idxs: {}; valid_idxs: {}".format(train_idxs, valid_idxs))
        train_all_dataset = CustomVisionDataset(train_all_dataset, train_idxs)
        valid_all_dataset = CustomVisionDataset(valid_all_dataset, valid_idxs)
        print("label_num: {}".format(label_num))
    # step = int(np.floor(num_train / SPLIT_NUM))
    # list_of_index = [train_idx[i: i + step] for i in range(0, len(train_idx), step)]
    
    list_of_index = dirichlet_split_noniid(train_all_dataset, ALPHA, SPLIT_NUM, same_capacity, plot_path)

    print("train_all_dataset_length: {}".format(len(train_all_dataset)))
    sub_train_datasets = []
    for indexes in list_of_index:
        if not has_origin_valid_dataset[name]:
            target_dataset = CustomVisionDataset(all_dataset, indexes)
        else:
            target_dataset = CustomVisionDataset(train_all_dataset, indexes)
        print("in dilikelei: {}".format(Counter(target_dataset.get_subset_targets())))
        sub_train_datasets.append(target_dataset)
    return train_all_dataset, sub_train_datasets, valid_all_dataset, label_num

'''
def get_CIFAR_dataset_mutli_split(VALID_SIZE, BATCH_SIZE, SPLIT_NUM):
    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    DATA_ROOT = '/home/linchangxiao/labInDiWu/dataset/cifar10'
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    all_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    num_all = len(all_dataset)
    label_num = len(set(all_dataset.classes))
    indices = list(range(num_all))
    num_train = int(np.floor((1 - VALID_SIZE) * num_all))
    num_valid = int(np.floor(VALID_SIZE * num_all))
    random.shuffle(indices)
    train_idx, valid_idx = indices[num_valid:], indices[:num_valid]
    
    step = int(np.floor(num_train / SPLIT_NUM))
    list_of_index = [train_idx[i: i + step] for i in range(0, len(train_idx), step)]
    # print('len(train_idx): {}; step: {}; len(list_of_index): {}'.format(len(train_idx), step, len(list_of_index)))
    train_loaders = []

    for index_li in list_of_index:
        
        # train_idx, valid_idx = index_li[split:], index_li[:split]
        train_sampler = SubsetRandomSampler(index_li)

        train_loader = DataLoader(
            all_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler
        )

        train_loaders.append(train_loader)

    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = DataLoader(
        all_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler
    )

    print("Finished Load Data!")
    return train_loaders, valid_loader, label_num


def get_CIFAR_dataset_mutli_split(VALID_SIZE, BATCH_SIZE, SPLIT_NUM):
    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    DATA_ROOT = '/home/linchangxiao/labInDiWu/dataset/cifar10'
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)
    valid_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    num_train = len(train_dataset)
    label_num = len(set(train_dataset.classes) | set(valid_dataset.classes))
    indices = list(range(num_train))
    random.shuffle(indices)
    step = int(np.floor(num_train / SPLIT_NUM))
    list_of_index = [indices[i: i + step] for i in range(0, len(indices), step)]
    
    train_loaders = []

    for index_li in list_of_index:
        # train_idx, valid_idx = index_li[split:], index_li[:split]
        train_sampler = SubsetRandomSampler(index_li)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler
        )

        train_loaders.append(train_loader)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
    )

    print("Finished Load Data!")
    return train_loaders, valid_loader, label_num
'''


def get_CIFAR_dataset(VALID_SIZE, BATCH_SIZE):
    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    DATA_ROOT = DATASET_PATH + '/cifar10'
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)
    valid_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(VALID_SIZE * num_train))
    # np.random.seed(RANDOM_SEED)
    # np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler
    )
    print("Finished Load Data!")
    return train_loader, valid_loader

def get_ImageNet_dataset(VALID_SIZE, BATCH_SIZE):
    DATA_ROOT = DATASET_PATH + '/imagenet/data/'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset_all = ImageFolder(DATA_ROOT, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    valid_dataset_all = ImageFolder(DATA_ROOT, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    num_train = len(train_dataset_all)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(VALID_SIZE * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_all,
        batch_size=BATCH_SIZE,
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset_all,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler
    )

    print("Finished Load Data!")
    return train_loader, valid_loader
