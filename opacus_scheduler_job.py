import warnings
warnings.simplefilter("ignore")

import torch
from torch.utils.data import ConcatDataset, DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
from utils.early_stop_tools import EarlyStopping
from utils.model_loader import get_PBS_LSTM, get_PBS_FF, get_pretained_Bert
from utils.data_loader import get_review_dataset_multi_split, get_review_dataset_combine_split
from utils.get_profiler_significance import get_profiler_significance_result
from utils.training_functions import privacy_model_train_valid
from utils.logging_tools import get_logger
from utils.global_variable import GLOBAL_PATH, DATASET_PATH, LOGGING_DATE
from utils.global_functions import add_2_map, normal_counter
import os
import contextlib
import json
from functools import reduce


def do_calculate_func(job_id, model_name, is_select, 
                    sub_train_datasets, valid_dataset,
                    selected_datablock_ids, not_selected_datablock_ids, 
                    device, early_stopping, summary_writer,
                    LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, 
                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, MAX_EPSILON, EPOCHS,
                    vocab_size, output_size, label_distributions,
                    train_configs, callback):
    begin_time = int(time.time())
    all_results = {}
    

    target_train_datasets = ConcatDataset([sub_train_datasets[id] for id in selected_datablock_ids])
    target_train_loader = DataLoader(target_train_datasets, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    selected_datablock_ids_str = '['
    for id in selected_datablock_ids:
        selected_datablock_ids_str += "{};".format(id)
    selected_datablock_ids_str += ']'
    select_log_str = "SELECT_{}".format(selected_datablock_ids_str)
    
    train_configs['vocab_size'] = vocab_size
    train_configs['output_size'] = output_size
    train_configs['current_datablock_idx'] = selected_datablock_ids
    train_configs['label_distributions'] = label_distributions
    
    train_acc, train_loss, valid_acc, valid_loss, epsilon_consume = privacy_model_train_valid(
        model_name, is_select, select_log_str, target_train_loader, valid_loader,
        device, output_size, early_stopping, summary_writer,
        LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, MAX_PHYSICAL_BATCH_SIZE, MAX_EPSILON, EPOCHS,
        train_configs
    )
    all_results[select_log_str] = {
        'train_acc': train_acc,
        'train_loss': train_loss,
        'valid_acc': valid_acc,
        'valid_loss': valid_loss,
        'epsilon_consume': epsilon_consume,
    }
    real_duration_time = int(time.time()) - begin_time
    callback(job_id, all_results, real_duration_time)