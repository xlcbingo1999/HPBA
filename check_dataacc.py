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
    
    out_path = "/home/netlab/DL_lab/opacus_testbed/dataacc.json"
    val_acc_epsilon_train_test = {}
    # with open(out_path, 'r') as f:
    #     val_acc_epsilon_train_test = json.load(f)
    for file in os.listdir(args.in_path):
        path = args.in_path + "/" + file
        dataset_name, model_name, epsilon, train_id, test_id, date = file.split("_")
        if model_name not in val_acc_epsilon_train_test:
            val_acc_epsilon_train_test[model_name] = {}
        if epsilon not in val_acc_epsilon_train_test[model_name]:
            val_acc_epsilon_train_test[model_name][epsilon] = {}
        if train_id not in val_acc_epsilon_train_test[model_name][epsilon]:
            val_acc_epsilon_train_test[model_name][epsilon][train_id] = {}
        if test_id not in val_acc_epsilon_train_test[model_name][epsilon][train_id]:
            val_acc_epsilon_train_test[model_name][epsilon][train_id][test_id] = {}
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
            val_acc_epsilon_train_test[model_name][epsilon][train_id][test_id][key] = last_value
            if key == "total_val_acc":
                index_10 = 9
                value_10 = df[key].iloc[index_10]
                val_acc_epsilon_train_test[model_name][epsilon][train_id][test_id]["{}_{}".format(key, index_10)] = value_10

    print(val_acc_epsilon_train_test)
    with open(out_path, 'w+') as f:
        json.dump(val_acc_epsilon_train_test, f)
    print("Tensorboard data exported successfully")

main()