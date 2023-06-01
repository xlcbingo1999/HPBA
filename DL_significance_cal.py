from utils.logging_tools import get_logger
from significance_policies.OTDD import OTDDPolicy
from significance_policies.Temp import TempPolicy

import time
import itertools
import torch

if __name__ == "__main__":
    # 测试
    torch.multiprocessing.set_start_method('spawn')

    simulation = False
    policy_name = "Temp"
    metric = "Accuracy"
    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    dispatcher_logger_path = f"/home/netlab/DL_lab/opacus_testbed/log_temp_store/{policy_name}_{current_time}_{metric}.log"
    logger = get_logger(dispatcher_logger_path, dispatcher_logger_path, enable_multiprocess=True)
    if "OTDD" in policy_name:
        policy = OTDDPolicy(simulation, logger)
    elif "Temp" in policy_name:
        policy = TempPolicy(simulation, metric, logger)

    train_datablock_num = 144
    type_id = dispatcher_logger_path
    
    train_dataset_names = ["EMNIST"]
    sub_train_key_ids = [f"train_sub_{index}" for index in range(train_datablock_num)]
    test_dataset_names = ["EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"] # "EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"
    sub_test_key_ids = ["test_sub_0"]
    model_names = ["CNN"] # 一个一个来 不然会出现延迟!
    
    batch_size = [1024]
    LR = [1e-3]
    
    temp_product_list = list(itertools.product(train_dataset_names, sub_train_key_ids, test_dataset_names, sub_test_key_ids, model_names, batch_size, LR))

    all_significance_state = [{
        "train_dataset_name": temp_product[0],
        "sub_train_key_id": temp_product[1],
        "test_dataset_name": temp_product[2],
        "sub_test_key_id": temp_product[3],
        "model_name": temp_product[4],
        "model_config": {
            "batch_size": temp_product[5],
            "LR": temp_product[6]
        }
    } for temp_product in temp_product_list]

    logger.info(f"len(all_significance_state): {len(all_significance_state)}")

    device_list = [0, 1, 2, 3] * 5 # 可以5组一起!
    policy.get_job_datablock_significance_async(type_id, all_significance_state, device_list)