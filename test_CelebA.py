import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, CenterCrop, Resize
from torchvision import models
import torch.nn.functional as F

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from utils.opacus_engine_tools import get_privacy_dataloader


import json
import time

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--EPSILON", type=float, default=10.0)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--train_id", type=int, default=0)
    parser.add_argument("--test_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="CNN") # resnet
    args = parser.parse_args()
    return args

def accuracy(preds, labels):
    return (preds == labels).mean()


args = get_df_config()

MODEL_NAME = args.model_name
EPOCHS = 50
DEVICE_INDEX = args.device_index
LR = 1e-3
EPSILON = args.EPSILON
DELTA = 1e-7
MAX_GRAD_NORM = 1.2

raw_data_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/CelebA'
sub_train_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/sub_train_datasets_config.json'
sub_test_config_path = '/mnt/linuxidc_client/dataset/Amazon_Review_split/test_dataset_config.json'

dataset_name = 'CelebA'
train_id = args.train_id
test_id = args.test_id
sub_train_key = 'train_sub_{}'.format(train_id)
sub_test_key = 'test_sub_{}'.format(test_id)

current_time =  time.strftime('%Y-%Y-%m-%d-%H-%M-%S', time.localtime())
summary_writer_path = '/mnt/linuxidc_client/tensorboard_20230305/CelebA_{}_{}_{}_{}_{}'.format(MODEL_NAME, EPSILON, train_id, test_id, current_time)

# with open(sub_train_config_path, 'r+') as f:
#     current_subtrain_config = json.load(f)
#     f.close()
# with open(sub_test_config_path, 'r+') as f:
#     current_subtest_config = json.load(f)
#     f.close()
# real_train_index = current_subtrain_config[dataset_name][sub_train_key]["indexes"]
# real_test_index = current_subtest_config[dataset_name][sub_test_key]["indexes"]

if MODEL_NAME == "CNN":
    transform = Compose([
        CenterCrop((178, 178)),
        Resize((128, 128)),
        ToTensor()
    ])
    BATCH_SIZE = 512
    MAX_PHYSICAL_BATCH_SIZE = int(BATCH_SIZE / 2)
elif MODEL_NAME == "resnet":
    transform = Compose([
        CenterCrop((178, 178)),
        Resize((128, 128)),
        ToTensor()
    ])
    BATCH_SIZE = 64
    MAX_PHYSICAL_BATCH_SIZE = 64

train_dataset = CelebA(
    root=raw_data_path,
    split="train",
    target_type="attr",
    download=False,
    transform=transform
)
test_dataset = CelebA(
    root=raw_data_path,
    split="test",
    target_type="attr",
    download=False,
    transform=transform
)

print("Finished load datasets!")
print("train num: {}".format(len(train_dataset)) )
print("test num: {}".format(len(test_dataset)) )

class CustomCelebADataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, target_attr_name, indices):
        assert len(dataset.target_type) == 1 and dataset.target_type[0] == "attr"
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.target_attr_name = target_attr_name
        self.target_attr_index = self.get_attr_index(target_attr_name)
        
    def __len__(self):
        return len(self.indices)

    def get_attr_index(self, target_attr_name):
        return self.dataset.attr_names.index(target_attr_name)

    def __getitem__(self, index):
        x, _ = self.dataset[self.indices[index]]
        target = []
        target.append(self.dataset.attr[self.indices[index], self.target_attr_index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.dataset.target_transform is not None:
                target = self.dataset.target_transform(target)
        else:
            target = None
        return x, target

    def get_target_distribution(self):
        all_indices_attr = self.dataset.attr[self.indices, self.target_attr_index]
        keys, values = all_indices_attr.unique(return_counts=True)
        sub_train_distribution = {
            str(target.item()): 0 for target in keys
        }
        for index in range(len(keys)):
            target = keys[index]
            num = values[index]
            sub_train_distribution[str(target.item())] = num.item()
        return sub_train_distribution

class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 128, 128)
            nn.Conv2d(
                in_channels=3,  # 输入通道数
                out_channels=16,  # 输出通道数
                kernel_size=5,   # 卷积核大小
                stride=1,  #卷积步数
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, 
                            # padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 128, 128)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 64, 64)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 64, 64)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 64, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(4),  # output shape (32, 16, 16)
        )
        self.out = nn.Linear(32 * 16 * 16, output_dim)  # 全连接层，A/Z,a/z一共37个类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 32 * 32)
        out = self.out(x)
        return out

print("begin train: {} test: {}".format(train_id, test_id))
train_dataset = CustomCelebADataset(train_dataset, "Male", range(len(train_dataset)))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_dataset = CustomCelebADataset(test_dataset, "Male", range(len(test_dataset)))
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
print("Finished split datasets!")
print("check train_loader: {}".format(len(train_loader) * BATCH_SIZE))
print("check test_loader: {}".format(len(test_loader) * BATCH_SIZE))


device = torch.device("cuda:{}".format(DEVICE_INDEX) if torch.cuda.is_available() else "cpu")

if MODEL_NAME == "CNN":
    model = CNN(output_dim=2)
elif MODEL_NAME == "resnet":
    model = models.resnet18(num_classes=2)
model = ModuleValidator.fix(model)
errors = ModuleValidator.validate(model, strict=False)
print("error: {}".format(errors))

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters


privacy_engine = PrivacyEngine() if EPSILON > 0.0 else None
model, optimizer, train_loader = \
    get_privacy_dataloader(privacy_engine, model, optimizer, 
                            train_loader, EPOCHS, 
                            EPSILON, DELTA, MAX_GRAD_NORM) 

summary_writer = SummaryWriter(summary_writer_path)
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = []
    total_train_acc = []
    # temp_debug_tensor = torch.zeros(size=(len(train_dataset.classes), ))
    if privacy_engine is not None:
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for i, (inputs, labels) in enumerate(memory_safe_data_loader):
                # temp_dis = labels.unique(return_counts=True)
                # temp_key = temp_dis[0]
                # temp_value = temp_dis[1]
                # for index in range(len(temp_key)):
                #     temp_debug_tensor[temp_key[index]] += temp_value[index]
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
                if (i + 1) % 10 == 0:
                    print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                    print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)))
                    # print("epoch[{}] check temp_debug_tensor: {}".format(epoch, temp_debug_tensor))
                    
    else:
        for i, (inputs, labels) in enumerate(train_loader):
            # print("check inputs: {}, labels: {}".format(inputs, labels))
            # temp_dis = labels.unique(return_counts=True)
            # temp_key = temp_dis[0]
            # temp_value = temp_dis[1]
            # for index in range(len(temp_key)):
            #     temp_debug_tensor[temp_key[index]] += temp_value[index]
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
            if (i + 1) % 10 == 0:
                print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                print("epoch[{}]: temp_train_acc: {}".format(epoch, np.mean(total_train_acc)))
                # print("epoch[{}] check temp_debug_tensor: {}".format(epoch, temp_debug_tensor))
    
    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(DELTA)
    else:
        epsilon = 0.0
    print("epoch[{}]: total_train_loss: {}".format(epoch, np.mean(total_train_loss)))
    print("epoch[{}]: total_train_acc: {}".format(epoch, np.mean(total_train_acc)))
    print("epoch[{}]: epsilon_consume: {}".format(epoch, epsilon))
    summary_writer.add_scalar('total_train_loss', np.mean(total_train_loss), epoch)
    summary_writer.add_scalar('total_train_acc', np.mean(total_train_acc), epoch)
    summary_writer.add_scalar('epsilon_consume', epsilon, epoch)

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
    summary_writer.add_scalar('total_val_loss', np.mean(total_val_loss), epoch)
    summary_writer.add_scalar('total_val_acc', np.mean(total_val_acc), epoch)

time.sleep(5)