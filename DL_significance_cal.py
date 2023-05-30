from utils.logging_tools import get_logger
from significance_policies.OTDD import OTDDPolicy

import time
import itertools

if __name__ == "__main__":
    # 测试
    simulation = False
    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    dispatcher_logger_path = f"/home/netlab/DL_lab/opacus_testbed/log_temp_store/OTDD_{current_time}.log"
    logger = get_logger(dispatcher_logger_path, dispatcher_logger_path, enable_multiprocess=True)
    policy = OTDDPolicy(simulation, logger)

    train_datablock_num = 144
    type_id = dispatcher_logger_path
    train_dataset_names = ["EMNIST"]
    sub_train_key_ids = [f"train_sub_{index}" for index in range(train_datablock_num)]
    test_dataset_names = ["EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"] # "EMNIST_MNIST-1000_1000", "EMNIST-2000", "MNIST-2000"
    sub_test_key_ids = ["test_sub_0"]
    
    temp_product_list = list(itertools.product(train_dataset_names, sub_train_key_ids, test_dataset_names, sub_test_key_ids))

    all_significance_state = [{
        "train_dataset_name": temp_product[0],
        "sub_train_key_id": temp_product[1],
        "test_dataset_name": temp_product[2],
        "sub_test_key_id": temp_product[3]
    } for temp_product in temp_product_list]

    logger.info(f"len(all_significance_state): {len(all_significance_state)}")

    device_list = [0, 1, 2, 3] * 15 # 可以六组一起!
    policy.get_job_datablock_significance_async(type_id, all_significance_state, device_list)