import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils.opacus_engine_tools import get_privacy_dataloader
from utils.split_dataset import split_and_draw


def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--EPSILON", type=float, default=15.0)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()
    return args

def accuracy(preds, labels):
    return (preds == labels).mean()


args = get_df_config()

BATCH_SIZE = 2048
MAX_PHYSICAL_BATCH_SIZE = int(BATCH_SIZE / 2)
EPOCHS = 50
DEVICE_INDEX = args.device_index
LR = 1e-3
EPSILON = args.EPSILON
DELTA = 1e-7
MAX_GRAD_NORM = 1.2
TRAIN_N_CLIENTS = 10
TEST_TYPES = 6
TRAIN_DIRICHLET_ALPHA = 1.0
TEST_DIRICHLET_ALPHA = 1.0

raw_data_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/EMNIST'
logger_path_prefix = '/home/netlab/DL_lab/opacus_testbed/log_20230214/EMNIST_{}_{}_{}_{}_{}'.format(EPSILON, TRAIN_N_CLIENTS, TEST_TYPES, TRAIN_DIRICHLET_ALPHA, TEST_DIRICHLET_ALPHA)
summary_writer_path = '/home/netlab/DL_lab/opacus_testbed/tensorboard_20230304/EMNIST_{}_{}_{}_{}_{}'.format(EPSILON, TRAIN_N_CLIENTS, TEST_TYPES, TRAIN_DIRICHLET_ALPHA, TEST_DIRICHLET_ALPHA)
summary_writer = SummaryWriter(summary_writer_path)
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

print("Finished load datasets!")
print("train num: {}; train class num: {}".format(len(train_dataset), len(train_dataset.classes)) )
print("test num: {}; test class num: {}".format(len(test_dataset), len(test_dataset.classes)) )



class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 输入通道数
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



train_dataset_size = len(train_dataset)
real_train_indices = split_and_draw("train", train_dataset, TRAIN_DIRICHLET_ALPHA, TRAIN_N_CLIENTS)
test_dataset_size = len(test_dataset)
real_test_indices = split_and_draw("test", test_dataset, TEST_DIRICHLET_ALPHA, TEST_TYPES)

for train_id, real_train_index in enumerate(real_train_indices):
    for test_id, read_test_index in enumerate(real_test_indices):
        print("begin train: {} test: {}".format(train_id, test_id))
        train_sampler = SubsetRandomSampler(real_train_index)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_sampler = SubsetRandomSampler(read_test_index)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
        print("Finished split datasets!")
        print("check train_loader: {}".format(len(train_loader) * BATCH_SIZE))
        print("check test_loader: {}".format(len(test_loader) * BATCH_SIZE))


        device = torch.device("cuda:{}".format(DEVICE_INDEX) if torch.cuda.is_available() else "cpu")

        model = CNN(output_dim=len(train_dataset.classes))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters


        privacy_engine = PrivacyEngine() if EPSILON > 0 else None
        model, optimizer, train_loader = \
            get_privacy_dataloader(privacy_engine, model, optimizer, 
                                    train_loader, EPOCHS, 
                                    EPSILON, DELTA, MAX_GRAD_NORM) 


        for epoch in range(EPOCHS):
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
                            print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                            print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)))
                            
            else:
                for i, (inputs, labels) in enumerate(train_loader):
                    # print("check inputs: {}, labels: {}".format(inputs, labels))
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
                        print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                        print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)))
            print("epoch[{}]: total_train_loss: {}".format(epoch, np.mean(total_train_loss)))
            print("epoch[{}]: total_train_acc: {}".format(epoch, np.mean(total_train_acc)))
            summary_writer.add_scalar('{}-{}/total_train_loss'.format(train_id, test_id), np.mean(total_train_loss), epoch)
            summary_writer.add_scalar('{}-{}/total_train_acc'.format(train_id, test_id), np.mean(total_train_acc), epoch)

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
                if (i + 1) % 1000 == 0:
                    print("val epoch[{}]: temp_val_loss: {}".format(epoch, np.mean(total_val_loss)))
                    print("val epoch[{}]: temp_val_acc: {}".format(epoch, np.mean(total_val_acc)))
            print("val epoch[{}]: total_val_loss: {}".format(epoch, np.mean(total_val_loss)))
            print("val epoch[{}]: total_val_acc: {}".format(epoch, np.mean(total_val_acc)))
            summary_writer.add_scalar('{}-{}/total_val_loss'.format(train_id, test_id), np.mean(total_val_loss), epoch)
            summary_writer.add_scalar('{}-{}/total_val_acc'.format(train_id, test_id), np.mean(total_val_acc), epoch)