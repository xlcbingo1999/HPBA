from utils.logging_tools import get_logger
from significance_policies.OTDD import OTDDPolicy
from significance_policies.Temp import TempPolicy

import time
import itertools
import torch
import logging

if __name__ == "__main__":
    cal_configs = [
        {
            "policy_name": "OTDD",
            "device_par": 5,
            "otdd_batch_size": 16,
            
            "temp_model_names": [],
            "temp_metric": None
        }, {
            "policy_name": "Temp",
            "device_par": 10,
            "otdd_batch_size": None,
            
            "temp_model_names": ["FF"],
            "temp_metric": "Accuracy"
        }, {
            "policy_name": "Temp",
            "device_par": 5,
            "otdd_batch_size": None,
            
            "temp_model_names": ["CNN"],
            "temp_metric": "Accuracy"
        }
    ]
    train_datablock_num = 48
    dataset_name = "EMNIST"
    dataset_config_name = f"subtrain_{train_datablock_num}_split_1.0_dirichlet"
    type_id = "No_importance"
    train_dataset_names = ["EMNIST"]
    sub_train_key_ids = [f"train_sub_{index}" for index in range(train_datablock_num)]
    test_dataset_names = ["EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"] # "EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"
    sub_test_key_ids = ["test_sub_0"]
    batch_size = [1024]
    LR = [1e-3]

    # 测试
    torch.multiprocessing.set_start_method('spawn')

    for cal_config in cal_configs:
        simulation = False
        policy_name = cal_config["policy_name"]
        
        current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())

        if "OTDD" in policy_name:
            dispatcher_logger_path = f"/home/netlab/DL_lab/opacus_testbed/log_temp_store/{policy_name}_{current_time}.log"
            logger = get_logger(dispatcher_logger_path, dispatcher_logger_path, enable_multiprocess=True, global_level=logging.INFO)
            cal_otdd_batch_size = cal_config["otdd_batch_size"]
            policy = OTDDPolicy(dataset_name, dataset_config_name, simulation, logger, batch_size=cal_otdd_batch_size)
            device_list = [0, 1, 2, 3] * cal_config["device_par"]
            model_names = ["No_importance"] # 一个一个来 不然会出现延迟!
        elif "Temp" in policy_name:
            metric = cal_config["temp_metric"]
            dispatcher_logger_path = f"/home/netlab/DL_lab/opacus_testbed/log_temp_store/{policy_name}_{current_time}_{metric}.log"
            logger = get_logger(dispatcher_logger_path, dispatcher_logger_path, enable_multiprocess=True, global_level=logging.INFO)
            policy = TempPolicy(dataset_name, dataset_config_name, metric, simulation, logger)
            device_list = [0, 1, 2, 3] * cal_config["device_par"] # CNN: 5 FF: 10
            model_names = cal_config["temp_model_names"] # 一个一个来 不然会出现延迟!
        
        
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

        
        policy.get_job_datablock_significance_async(type_id, all_significance_state, device_list)