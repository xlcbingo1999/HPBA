import os
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
import time

def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    
    for file in os.listdir(args.in_path):
        path = args.in_path + "/" + file
        event_data = event_accumulator.EventAccumulator(path)  # a python interface for loading Event data
        new_writer = SummaryWriter(args.out_path)
        
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        keys = event_data.scalars.Keys()  # get all tags,save in a list
        # print(keys)
        df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
        for key in tqdm(keys):
            print(key)
            df[key] = pd.DataFrame(event_data.Scalars(key)).value
            new_key = key.split("/")[1]
            for index, value in enumerate(df[key]):
                print("check key: {} index: {} value: {}".format(new_key, index, value))
                new_writer.add_scalar(new_key, value ,index)
                
        # print(df)
        # basename = os.path.basename(path)
        # ex_path = "/home/ubuntu/data/labInDiWu/cacheRL/output_csv/{}/".format(args.ex_path)
        # output_path = ex_path + basename + ".csv"
        # df.to_csv(output_path)
        
        time.sleep(3)
        print("Tensorboard data exported successfully")

main()