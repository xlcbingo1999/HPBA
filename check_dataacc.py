import os
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
import time
import json

def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in_path', type=str, required=True)
    # parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    
    val_acc_epsilon_train_test = {}
    for file in os.listdir(args.in_path):
        path = args.in_path + "/" + file
        dataset_name, epsilon, train_id, test_id, date = file.split("_")
        if epsilon not in val_acc_epsilon_train_test:
            val_acc_epsilon_train_test[epsilon] = {}
        if train_id not in val_acc_epsilon_train_test[epsilon]:
            val_acc_epsilon_train_test[epsilon][train_id] = {}
        if test_id not in val_acc_epsilon_train_test[epsilon][train_id]:
            val_acc_epsilon_train_test[epsilon][train_id][test_id] = {}
        event_data = event_accumulator.EventAccumulator(path)  # a python interface for loading Event data
        
        event_data.Reload()  # synchronously loads all of the data written so far b
        keys = event_data.scalars.Keys()  # get all tags,save in a list
        df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
        for key in tqdm(keys):
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
            t_index = -1
            last_value = df[key].iloc[t_index]
            while not pd.notna(last_value):
                last_value = df[key].iloc[t_index-1]
            val_acc_epsilon_train_test[epsilon][train_id][test_id][key] = last_value
    
    print(val_acc_epsilon_train_test)
    out_path = "/home/netlab/DL_lab/VolumeBased-DataValuation/dataacc_cnn.json"
    with open(out_path, 'w') as f:
        json.dump(val_acc_epsilon_train_test, f)
    print("Tensorboard data exported successfully")

main()