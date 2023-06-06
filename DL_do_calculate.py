# 目前是正常的
import argparse
import json
import zerorpc
import time
import sys
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

from utils.global_variable import DATASET_PATH
from utils.global_functions import print_console_file, get_zerorpc_client
from utils.data_loader import get_concat_dataset
from utils.model_loader import PrivacyCNN, PrivacyFF

import string
import os

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--worker_ip", type=str, required=True)
    parser.add_argument("--worker_port", type=str, required=True)
    
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_dataset_name", type=str, required=True) # : 用这个进行split
    parser.add_argument("--test_dataset_name", type=str, required=True)
    parser.add_argument("--sub_train_key_ids", type=str, required=True) # : 用这个进行split
    parser.add_argument("--sub_test_key_id", type=str, required=True)

    parser.add_argument("--sub_train_dataset_config_path", type=str, required=True)
    parser.add_argument("--test_dataset_config_path", type=str, required=True)
   
    parser.add_argument("--device_index", type=int, required=True)
    parser.add_argument("--logging_file_path", type=str, default="")
    parser.add_argument("--summary_writer_path", type=str, default="")
    parser.add_argument("--summary_writer_key", type=str, default="")
    parser.add_argument("--model_save_path", type=str, default="")
    parser.add_argument("--LR", type=float, required=True)
    parser.add_argument("--EPSILON_one_siton", type=float, required=True)
    parser.add_argument("--DELTA", type=float, required=True)
    parser.add_argument("--MAX_GRAD_NORM", type=float, required=True)
    parser.add_argument("--BATCH_SIZE", type=int, required=True)
    parser.add_argument("--MAX_PHYSICAL_BATCH_SIZE", type=int, required=True)
    parser.add_argument("--begin_epoch_num", type=int, required=True)
    parser.add_argument("--siton_run_epoch_num", type=int, required=True)

    parser.add_argument("--final_significance", type=float, required=True)
    parser.add_argument("--simulation_flag", action="store_true")
    args = parser.parse_args()
    return args

def accuracy(preds, labels):
    return (preds == labels).mean()

def do_calculate_func(job_id, model_name, 
                    train_dataset_name, sub_train_key_ids,
                    test_dataset_name, sub_test_key_id,
                    sub_train_dataset_config_path, test_dataset_config_path,
                    device_index,
                    model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
                    LR, EPSILON_one_siton, DELTA, MAX_GRAD_NORM, 
                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, 
                    begin_epoch_num, siton_run_epoch_num, final_significance, 
                    simulation_flag):
    begin_time = time.time()

    if simulation_flag:
        all_results = {
            'train_acc': 0.0,
            'train_loss': 0.0,
            'test_acc': 0.0,
            'test_loss': 0.0,
            'epsilon_consume': EPSILON_one_siton,
            'begin_epoch_num': begin_epoch_num,
            'siton_run_epoch_num': siton_run_epoch_num,
            'final_significance': final_significance
        }
        real_duration_time = time.time() - begin_time
        return job_id, all_results, real_duration_time
    
    with open(logging_file_path, "a+") as f:
        print_console_file("check train_dataset_name: {}".format(train_dataset_name), fileHandler=f)
        print_console_file("check sub_train_key_ids: {}".format(sub_train_key_ids), fileHandler=f)
        print_console_file("check test_dataset_name: {}".format(test_dataset_name), fileHandler=f)
        print_console_file("check sub_test_key_id: {}".format(sub_test_key_id), fileHandler=f)
        print_console_file("check device_index: {}".format(device_index), fileHandler=f)
        print_console_file("check final_significance: {}".format(final_significance), fileHandler=f)
        print_console_file("check EPSILON_one_siton: {}".format(EPSILON_one_siton), fileHandler=f)
        
    train_dataset = get_concat_dataset(train_dataset_name, sub_train_key_ids, 
                                    DATASET_PATH, sub_train_dataset_config_path, 
                                    "train")
    
    test_dataset = get_concat_dataset(test_dataset_name, sub_test_key_id,
                                    DATASET_PATH, test_dataset_config_path,
                                    "test")
    with open(logging_file_path, "a+") as f:
        print_console_file("finished load train_dataset", fileHandler=f)
        print_console_file("finished load test_dataset", fileHandler=f)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda:{}".format(device_index) if torch.cuda.is_available() else "cpu")
    if model_name == "CNN":
        model = PrivacyCNN(output_dim=len(train_dataset.classes))
    elif model_name == "FF":
        model = PrivacyFF(output_dim=len(train_dataset.classes))
    elif model_name == "resnet18":
        model = models.resnet18(num_classes=len(train_dataset.classes))
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    with open(logging_file_path, "a+") as f:
        print_console_file("finished load model and state_dict", fileHandler=f)
    model.train()

    if EPSILON_one_siton > 0.0:
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        with open(logging_file_path, "a+") as f:
            print_console_file("error: {}".format(errors), fileHandler=f)

    model = model.to(device)
    with open(logging_file_path, "a+") as f:
        print_console_file(f"model to device({device_index})", fileHandler=f)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    with open(logging_file_path, "a+") as f:
        print_console_file("finished criterion optimizer", fileHandler=f)
    
    privacy_engine = PrivacyEngine() if EPSILON_one_siton > 0.0 else None
    model, optimizer, train_loader = \
        get_privacy_dataloader(privacy_engine, model, optimizer, 
                                train_loader, siton_run_epoch_num, 
                                EPSILON_one_siton, DELTA, MAX_GRAD_NORM) 
    
    with open(logging_file_path, "a+") as f:
        print_console_file("job [{}] - epoch [{} to {}] begining ...".format(job_id, begin_epoch_num, begin_epoch_num + siton_run_epoch_num), fileHandler=f)
        
    summary_writer = SummaryWriter(summary_writer_path)
    for epoch in range(siton_run_epoch_num):
        model.train()
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
                    total_train_loss.append(loss.item())

                    preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                    labels = labels.detach().cpu().numpy()
                    acc = accuracy(preds, labels)
                    total_train_acc.append(acc)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 100 == 0:
                        with open(logging_file_path, "a+") as f:
                            print_console_file("epoch[{}]: temp_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)), fileHandler=f)
                            print_console_file("epoch[{}]: temp_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)), fileHandler=f)
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
                if (i + 1) % 100 == 0:
                    with open(logging_file_path, "a+") as f:
                        print_console_file("epoch[{}]: temp_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)), fileHandler=f)
                        print_console_file("epoch[{}]: temp_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)), fileHandler=f)
  
        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(DELTA)
        else:
            epsilon = 0.0
        with open(logging_file_path, "a+") as f:
            print_console_file("epoch[{}]: total_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)), fileHandler=f)
            print_console_file("epoch[{}]: total_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)), fileHandler=f)
            print_console_file("epoch[{}]: epsilon_consume: {}".format(begin_epoch_num + epoch, epsilon), fileHandler=f)
        summary_writer.add_scalar('{}/total_train_loss'.format(summary_writer_key), np.mean(total_train_loss), begin_epoch_num + epoch)
        summary_writer.add_scalar('{}/total_train_acc'.format(summary_writer_key), np.mean(total_train_acc), begin_epoch_num + epoch)
        summary_writer.add_scalar('{}/epsilon_consume'.format(summary_writer_key), epsilon, begin_epoch_num + epoch)

        model.eval()
        total_val_loss = []
        total_val_acc = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            total_val_loss.append(loss.item())

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            total_val_acc.append(acc)
            if (i + 1) % 10 == 0:
                with open(logging_file_path, "a+") as f:
                    print_console_file("val epoch[{}]: temp_val_loss: {}".format(begin_epoch_num + epoch, np.mean(total_val_loss)), fileHandler=f)
                    print_console_file("val epoch[{}]: temp_val_acc: {}".format(begin_epoch_num + epoch, np.mean(total_val_acc)), fileHandler=f)

        with open(logging_file_path, "a+") as f:
            print_console_file("val epoch[{}]: total_val_loss: {}".format(begin_epoch_num + epoch, np.mean(total_val_loss)), fileHandler=f)
            print_console_file("val epoch[{}]: total_val_acc: {}".format(begin_epoch_num + epoch, np.mean(total_val_acc)), fileHandler=f)
        summary_writer.add_scalar('{}/total_val_loss'.format(summary_writer_key), np.mean(total_val_loss), begin_epoch_num + epoch)
        summary_writer.add_scalar('{}/total_val_acc'.format(summary_writer_key), np.mean(total_val_acc), begin_epoch_num + epoch)
    
    summary_writer.close()
    if len(model_save_path) > 0:
        if not os.path.exists(model_save_path):
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model._module.state_dict(), model_save_path)

    real_duration_time = time.time() - begin_time
    all_results = {
        'train_acc': np.mean(total_train_acc),
        'train_loss': np.mean(total_train_loss),
        'test_acc': np.mean(total_val_acc),
        'test_loss': np.mean(total_val_loss),
        'epsilon_consume': epsilon,
        'begin_epoch_num': begin_epoch_num,
        'siton_run_epoch_num': siton_run_epoch_num,
        'final_significance': final_significance
    }

    with open(logging_file_path, "a+") as f:
        print_console_file("job [{}] saves in {}".format(job_id, model_save_path), fileHandler=f)
        print_console_file("job [{}] - epoch [{} to {}] end ".format(job_id, begin_epoch_num, begin_epoch_num + siton_run_epoch_num), fileHandler=f)

    return job_id, all_results, real_duration_time

if __name__ == "__main__":
    args = get_df_config()

    worker_ip = args.worker_ip 
    worker_port = args.worker_port 
    job_id = args.job_id 
    
    model_name = args.model_name

    train_dataset_name = args.train_dataset_name
    test_dataset_name = args.test_dataset_name

    sub_train_key_ids = args.sub_train_key_ids
    sub_train_key_ids = sub_train_key_ids.split(":")
    sub_test_key_id = args.sub_test_key_id

    sub_train_dataset_config_path = args.sub_train_dataset_config_path
    test_dataset_config_path = args.test_dataset_config_path

    device_index = args.device_index

    model_save_path = args.model_save_path
    summary_writer_path = args.summary_writer_path
    summary_writer_key = args.summary_writer_key
    logging_file_path = args.logging_file_path
    
    LR = args.LR 
    EPSILON_one_siton = args.EPSILON_one_siton
    DELTA = args.DELTA
    MAX_GRAD_NORM = args.MAX_GRAD_NORM 
    BATCH_SIZE = args.BATCH_SIZE
    MAX_PHYSICAL_BATCH_SIZE = args.MAX_PHYSICAL_BATCH_SIZE

    final_significance = args.final_significance
    simulation_flag = args.simulation_flag

    begin_epoch_num = args.begin_epoch_num
    siton_run_epoch_num = args.siton_run_epoch_num    
    try:
        job_id, all_results, real_duration_time = do_calculate_func(
            job_id, model_name, 
            train_dataset_name, sub_train_key_ids,
            test_dataset_name, sub_test_key_id,
            sub_train_dataset_config_path, test_dataset_config_path,
            device_index, 
            model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
            LR, EPSILON_one_siton, DELTA, MAX_GRAD_NORM, 
            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, 
            begin_epoch_num, siton_run_epoch_num, final_significance, 
            simulation_flag
        )
        
        with open(logging_file_path, "a+") as f:
            print_console_file(f"finished callback to worker: {worker_ip}:{worker_port}", fileHandler=f)
        client = get_zerorpc_client(worker_ip, worker_port)
        client.finished_job_callback(job_id, all_results, real_duration_time)
    except Exception as e:
        with open(logging_file_path, "a+") as f:
            print_console_file(f"runtime_failed callback to worker: {worker_ip}:{worker_port} with info {e}", fileHandler=f)
        client = get_zerorpc_client(worker_ip, worker_port)
        client.runtime_failed_job_callback(job_id, str(e))
    finally:
        print_console_file("finally finished!", fileHandler=f)
        time.sleep(5)
        sys.exit(0)