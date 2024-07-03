import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import dask.dataframe as dd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.tensorboard import SummaryWriter
import time


from utils.opacus_engine_tools import get_privacy_dataloader

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--EPSILON", type=float, default=15.0)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--BATCH_SIZE", type=int, default=8192)
    args = parser.parse_args()
    return args

class MLP(nn.Module):
    def __init__(self, in_features):
        super(MLP, self).__init__()
        # print("in_features: ", in_features)
        self.hidden1 = nn.Linear(in_features, 20)
        self.hidden2 = nn.Linear(20,20)
        self.hidden3 = nn.Linear(20,5)
        self.predict = nn.Linear(5,1)
    
    def forward(self, x):
        # print("check x.shape: ", x.shape)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        out = output.view(-1)
        return out

if __name__ == "__main__":
    args = get_df_config()
    device_index = args.device_index
    EPSILON = args.EPSILON
    sample_frac = args.sample_frac
    BATCH_SIZE = args.BATCH_SIZE
    MAX_PHYSICAL_BATCH_SIZE = int(BATCH_SIZE/2)
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.2
    lr = 1e-2
    EPOCHS = 50

    device = torch.device("cuda:{}".format(device_index) if torch.cuda.is_available() else "cpu")
    print(device)
    dataset_path = "/mnt/linuxidc_client/dataset/Amazon_Review_split/NYC_taxi_dataset/yellow_tripdata_2019-01-all.csv"
    summary_writer_path = "/home/netlab/DL_lab/opacus_testbed/tensorboard_nyc"
    summary_writer = SummaryWriter(summary_writer_path)
    model_name = "MLP"
    dataset_name = "nyc"
    summary_writer_date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    summary_writer_keyword = "{}-{}-{}".format(model_name, dataset_name, summary_writer_date)
    df = pd.read_csv(dataset_path)
    if 0.0 < sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac)
    print("Finished load dataset!")

    df_feature = df.drop(
        columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "tip_amount", "store_and_fwd_flag", "fare_amount"]
    )
    df_label = df["fare_amount"]

    df_feature_np = np.array(df_feature)
    df_label_np = np.array(df_label)
    X_train, X_test, y_train, y_test = train_test_split(df_feature_np, df_label_np, test_size=0.1)

    X_train = torch.from_numpy(X_train).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)

    print("check X_train: ", X_train.dtype)
    print("check X_test: ", X_test.dtype)
    print("check y_train: ", y_train.dtype)
    print("check y_test: ", y_test.dtype)


    all_train_dataset = TensorDataset(
        X_train,
        y_train
    )
    test_dataset = TensorDataset(
        X_test,
        y_test
    )
    train_loader = DataLoader(all_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    in_features = X_train.shape[1]
    model = MLP(in_features)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    privacy_engine = PrivacyEngine() if EPSILON > 0 else None
    model, optimizer, train_loader = \
        get_privacy_dataloader(privacy_engine, model, optimizer, 
                                train_loader, EPOCHS, 
                                EPSILON, DELTA, MAX_GRAD_NORM) 

    model.train()
    for epoch in range(EPOCHS):
        total_train_loss = []
        total_train_mae = []
        if privacy_engine is not None:
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for i, (inputs, labels) in enumerate(memory_safe_data_loader):
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output = model(inputs)
                    loss = criterion(output, labels)
                        
                    total_train_loss.append(loss.item())
                    train_mae = mean_absolute_error(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
                    total_train_mae.append(train_mae)
                    loss.backward()
                    optimizer.step()
                    summary_writer.add_scalar('{}/train_loss'.format(summary_writer_keyword), np.mean(total_train_loss), epoch)
                    summary_writer.add_scalar('{}/train_mae'.format(summary_writer_keyword), np.mean(total_train_mae), epoch)
                    epsilon_consume = privacy_engine.get_epsilon(DELTA)
                    summary_writer.add_scalar('{}/all_epsilon_consume'.format(summary_writer_keyword), epsilon_consume, epoch)
                    if (i + 1) % 1000 == 0:
                        print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                        print("epoch[{}]: temp_train_mae: {}".format(epoch, np.mean(total_train_mae)))
        else:
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                loss = criterion(output, labels)
                    
                total_train_loss.append(loss.item())
                train_mae = mean_absolute_error(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
                total_train_mae.append(train_mae)
                loss.backward()
                optimizer.step()
                summary_writer.add_scalar('{}/train_loss'.format(summary_writer_keyword), np.mean(total_train_loss), epoch)
                summary_writer.add_scalar('{}/train_mae'.format(summary_writer_keyword), np.mean(total_train_mae), epoch)
                epsilon_consume = 0.0
                summary_writer.add_scalar('{}/all_epsilon_consume'.format(summary_writer_keyword), epsilon_consume, epoch)
                if (i + 1) % 1000 == 0:
                    print("epoch[{}]: temp_train_loss: {}".format(epoch, np.mean(total_train_loss)))
                    print("epoch[{}]: temp_train_mae: {}".format(epoch, np.mean(total_train_mae)))
        print("epoch[{}]: total_train_loss: {}".format(epoch, np.mean(total_train_loss)))
        print("epoch[{}]: total_train_mae: {}".format(epoch, np.mean(total_train_mae)))

    model.eval()
    total_val_loss = []
    total_val_mae = []
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
            
        total_val_loss.append(loss.item())
        val_mae = mean_absolute_error(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
        total_val_mae.append(val_mae)
        if (i + 1) % 1000 == 0:
            print("val: temp_val_loss: {}".format(np.mean(total_val_loss)))
            print("val: temp_val_mae: {}".format(np.mean(total_val_mae)))
    print("val: total_val_loss: {}".format(np.mean(total_val_loss)))
    print("val: total_val_mae: {}".format(np.mean(total_val_mae)))

