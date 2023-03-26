'''
import argparse
from utils.global_variable import WORKER_LOCAL_IP, WORKER_LOCAL_PORT
from significance_policies.OTDD import OTDDPolicy

def get_df_config():
    parser = argparse.ArgumentParser(
                description="Sweep through lambda values")
    parser.add_argument("--train_dataset_name", type=str, default="EMNIST")
    parser.add_argument("--test_dataset_name", type=str, default="EMNIST_MNIST-1000_1000") # MNIST-2000, EMNIST-2000
    parser.add_argument("--sub_train_key_ids", type=str, default="train_sub_0")
    parser.add_argument("--sub_test_key_ids", type=str, default="test_sub_0")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    return args

args = get_df_config()
ot = OTDDPolicy(args.batch_size)
signficance_state = {
    "train_dataset_name": args.train_dataset_name,
    "test_dataset_name": args.test_dataset_name,
    "sub_train_key_ids": [args.sub_train_key_ids],
    "sub_test_key_ids": [args.sub_test_key_ids],
}
ot.get_job_datablock_significance_async(signficance_state, args.device_index)
'''

import os
path = '/mnt/linuxidc_client/models_save'
subpath = path + '/d20230324'
file = subpath + '/d.txt'

if not os.path.exists(file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write("File created successfully!")
else:
    print("File already exists!")