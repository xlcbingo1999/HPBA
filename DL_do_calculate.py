# 目前是正常的
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
   
    parser.add_argument("--device_index", type=int, required=True)
    parser.add_argument("--logging_file_path", type=str, default="")
    parser.add_argument("--summary_writer_path", type=str, default="")
    parser.add_argument("--summary_writer_key", type=str, default="")
    parser.add_argument("--model_save_path", type=str, default="")
    parser.add_argument("--LR", type=float, required=True)
    parser.add_argument("--EPSILON", type=float, required=True)
    parser.add_argument("--DELTA", type=float, required=True)
    parser.add_argument("--MAX_GRAD_NORM", type=float, required=True)
    parser.add_argument("--BATCH_SIZE", type=int, required=True)
    parser.add_argument("--MAX_PHYSICAL_BATCH_SIZE", type=int, required=True)
    parser.add_argument("--begin_epoch_num", type=int, required=True)
    parser.add_argument("--run_epoch_num", type=int, required=True)

    args = parser.parse_args()
    return args

def accuracy(preds, labels):
    return (preds == labels).mean()

class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,  # 输入通道数
                out_channels=16,  # 输出通道数
                kernel_size=5,   # 卷积核大小
                stride=1,  #卷积步数
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, 
                            # padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, output_dim)  # 全连接层，A/Z,a/z一共37个类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

class FF(nn.Module):
    def __init__(self, output_dim):
        super(FF, self).__init__()
        self.hidden = [128, 32]
        self.linear = nn.Sequential(
            nn.Linear(28 * 28, self.hidden[0], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden[0], self.hidden[1], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden[1], output_dim, bias=True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output    

def do_calculate_func(job_id, model_name, 
                    train_dataset_name, sub_train_key_ids,
                    test_dataset_name, sub_test_key_id,
                    device_index,
                    model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
                    LR, EPSILON, DELTA, MAX_GRAD_NORM, 
                    BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, 
                    begin_epoch_num, run_epoch_num):
    begin_time = time.time()
    
    with open(logging_file_path, "a+") as f:
        print("check train_dataset_name: {}".format(train_dataset_name), file=f)
        print("check sub_train_key_ids: {}".format(sub_train_key_ids), file=f)
        print("check test_dataset_name: {}".format(test_dataset_name), file=f)
        print("check sub_test_key_id: {}".format(sub_test_key_id), file=f)
        print("check device_index: {}".format(device_index), file=f)
        
    train_dataset = get_concat_dataset(train_dataset_name, sub_train_key_ids, 
                                    DATASET_PATH, SUB_TRAIN_DATASET_CONFIG_PATH, 
                                    "train")
    test_dataset = get_concat_dataset(test_dataset_name, sub_test_key_id,
                                    DATASET_PATH, TEST_DATASET_CONFIG_PATH,
                                    "test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda:{}".format(device_index) if torch.cuda.is_available() else "cpu")
    if model_name == "CNN":
        model = CNN(output_dim=len(train_dataset.classes))
    elif model_name == "FF":
        model = CNN(output_dim=len(train_dataset.classes))
    elif model_name == "resnet18":
        model = models.resnet18(num_classes=len(train_dataset.classes))
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    model.train()

    if EPSILON > 0.0:
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        print("error: {}".format(errors))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters

    
    privacy_engine = PrivacyEngine() if EPSILON > 0.0 else None
    model, optimizer, train_loader = \
        get_privacy_dataloader(privacy_engine, model, optimizer, 
                                train_loader, run_epoch_num, 
                                run_epoch_num * EPSILON, DELTA, MAX_GRAD_NORM) 

    with open(logging_file_path, "a+") as f:
        print("job [{}] - epoch [{} to {}] begining ...".format(job_id, begin_epoch_num, begin_epoch_num + run_epoch_num))
        print("job [{}] - epoch [{} to {}] begining ...".format(job_id, begin_epoch_num, begin_epoch_num + run_epoch_num), file=f)
        
    summary_writer = SummaryWriter(summary_writer_path)
    for epoch in range(run_epoch_num):
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
                            print("epoch[{}]: temp_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)), file=f)
                            print("epoch[{}]: temp_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)), file=f)
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
                        print("epoch[{}]: temp_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)), file=f)
                        print("epoch[{}]: temp_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)), file=f)
  
        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(DELTA)
        else:
            epsilon = 0.0
        with open(logging_file_path, "a+") as f:
            print("epoch[{}]: total_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)))
            print("epoch[{}]: total_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)))
            print("epoch[{}]: epsilon_consume: {}".format(begin_epoch_num + epoch, epsilon))
            print("epoch[{}]: total_train_loss: {}".format(begin_epoch_num + epoch, np.mean(total_train_loss)), file=f)
            print("epoch[{}]: total_train_acc: {}".format(begin_epoch_num + epoch, np.mean(total_train_acc)), file=f)
            print("epoch[{}]: epsilon_consume: {}".format(begin_epoch_num + epoch, epsilon), file=f)
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
                    print("val epoch[{}]: temp_val_loss: {}".format(begin_epoch_num + epoch, np.mean(total_val_loss)), file=f)
                    print("val epoch[{}]: temp_val_acc: {}".format(begin_epoch_num + epoch, np.mean(total_val_acc)), file=f)

        with open(logging_file_path, "a+") as f:
            print("val epoch[{}]: total_val_loss: {}".format(begin_epoch_num + epoch, np.mean(total_val_loss)))
            print("val epoch[{}]: total_val_acc: {}".format(begin_epoch_num + epoch, np.mean(total_val_acc)))
            print("val epoch[{}]: total_val_loss: {}".format(begin_epoch_num + epoch, np.mean(total_val_loss)), file=f)
            print("val epoch[{}]: total_val_acc: {}".format(begin_epoch_num + epoch, np.mean(total_val_acc)), file=f)
        summary_writer.add_scalar('{}/total_val_loss'.format(summary_writer_key), np.mean(total_val_loss), begin_epoch_num + epoch)
        summary_writer.add_scalar('{}/total_val_acc'.format(summary_writer_key), np.mean(total_val_acc), begin_epoch_num + epoch)
    
    summary_writer.close()
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
        'run_epoch_num': run_epoch_num
    }

    with open(logging_file_path, "a+") as f:
        print("job [{}] - epoch [{} to {}] end ".format(job_id, begin_epoch_num, begin_epoch_num + run_epoch_num))

        print("job [{}] saves in {}".format(job_id, model_save_path), file=f)
        print("job [{}] - epoch [{} to {}] end ".format(job_id, begin_epoch_num, begin_epoch_num + run_epoch_num), file=f)

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

    device_index = args.device_index

    summary_writer_path = args.summary_writer_path
    summary_writer_key = args.summary_writer_key
    logging_file_path = args.logging_file_path
    model_save_path = args.model_save_path
    
    LR = args.LR 
    EPSILON = args.EPSILON
    DELTA = args.DELTA
    MAX_GRAD_NORM = args.MAX_GRAD_NORM 
    BATCH_SIZE = args.BATCH_SIZE
    MAX_PHYSICAL_BATCH_SIZE = args.MAX_PHYSICAL_BATCH_SIZE

    begin_epoch_num = args.begin_epoch_num
    run_epoch_num = args.run_epoch_num

    job_id, all_results, real_duration_time = do_calculate_func(
        job_id, model_name, 
        train_dataset_name, sub_train_key_ids,
        test_dataset_name, sub_test_key_id,
        device_index, 
        model_save_path, summary_writer_path, summary_writer_key, logging_file_path,
        LR, EPSILON, DELTA, MAX_GRAD_NORM, 
        BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, 
        begin_epoch_num, run_epoch_num
    )
    
    tcp_ip_port = "tcp://{}:{}".format(worker_ip, worker_port)
    client = zerorpc.Client()
    client.connect(tcp_ip_port)
    client.finished_job_callback(job_id, all_results, real_duration_time)