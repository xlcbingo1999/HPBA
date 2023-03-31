import numpy as np
import random
from utils.global_variable import SCHE_IP, SCHE_PORT, DISPATCHER_IP, DISPATCHER_PORT, INIT_WORKERIDENTIFIERS, RESULT_PATH


def generate_dataset(dataset_names, fix_epsilon=10.0, fix_delta=1e-5, fix_time=0, num=6):
    print("check dataset_names: {}".format(dataset_names))
    enable_train_dataset_names = ["EMNIST"]
    # enable_test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]
    result = {}
    for name in dataset_names:
        if name not in enable_train_dataset_names:
            continue
        result[name] = {}
        for index in range(num):
            sub_datablock_name = "train_sub_{}".format(index)
            result[name][sub_datablock_name] = {
                "submited": False,
                "epsilon_capacity": fix_epsilon,
                "delta_capacity": fix_delta,
                "time": fix_time
            }
        print(result[name])
    return result

def generate_normal_one_job(time, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA, 
                            update_sched_epoch_num, TARGET_EPOCHS, dispatcher_ip, dispatcher_port,
                            is_history=False):
    if (is_history and time > 0) or (not is_history and time < 0):
        time = -time
    job_detail = {
        "time": time,
        "model_name": model_name,
        "train_dataset_name": train_dataset_name,
        "test_dataset_name": test_dataset_name,
        "sub_test_key_id": "test_sub_0",
        "datablock_select_num": datablock_select_num,
        "LR": 1e-3,
        "EPSILON": EPSILON,
        "DELTA": DELTA,
        "update_sched_epoch_num": update_sched_epoch_num,
        "MAX_EPOCHS": TARGET_EPOCHS * 2,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_PHYSICAL_BATCH_SIZE": MAX_PHYSICAL_BATCH_SIZE,
        "TARGET_EPOCHS": TARGET_EPOCHS,
        "priority_weight": 1.0,
        "dispatcher_ip": dispatcher_ip,
        "dispatcher_port": dispatcher_port,
    }
    if not is_history:
        job_detail["submited"] = False
    return job_detail

def poisson_arrival_times(last_arrival_time, lambdas):
    # n: 总任务数
    # lambdas: 每个任务的到达率
    arrival_time = last_arrival_time + np.random.exponential(scale=1/lambdas)
    return arrival_time

def generate_jobs(all_decision_num, update_sched_epoch_num, per_epoch_EPSILONs, EPSILONs_weights, is_history):
    # 在这里应该生成比较多的类型
    # all_big_job_num = int(all_decision_num / update_sched_epoch_num)
    # lambdas = [random.random() / 10 for _ in range(all_big_job_num)]
    # all_big_job_arrival_times = poisson_arrival_times(all_big_job_num, lambdas)
    
    models = ["CNN"]
    BATCH_SIZEs = [1024]
    MAX_PHYSICAL_BATCH_SIZEs = [512]
    TARGET_EPOCHSs = [100]
     
    train_dataset_names = ["EMNIST"]
    test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]
    test_dataset_names_weights = [0.8, 0.15, 0.05]
    datablock_select_nums = [1, 2, 4, 8]
    datablock_select_nums_weights = [0.6, 0.2, 0.15, 0.05]
    
    jobs = []
    current_decision_num = 0
    last_arrival_time = 0.0
    while current_decision_num < all_decision_num:
        model_name_index_list = [i for i, _ in enumerate(models)]
        model_name_i = random.choices(model_name_index_list)[0]
        model_name = models[model_name_i]
        
        train_dataset_name = random.choices(train_dataset_names)[0]

        test_dataset_names_index_list = [i for i, _ in enumerate(test_dataset_names)]
        test_dataset_name_i = random.choices(test_dataset_names_index_list, weights=test_dataset_names_weights)[0]
        test_dataset_name = test_dataset_names[test_dataset_name_i]

        datablock_select_nums_index_list = [i for i, _ in enumerate(datablock_select_nums)]
        datablock_select_num_i = random.choices(datablock_select_nums_index_list, weights=datablock_select_nums_weights)[0]
        datablock_select_num = datablock_select_nums[datablock_select_num_i]

        BATCH_SIZE = random.choices(BATCH_SIZEs)[0]
        MAX_PHYSICAL_BATCH_SIZE = random.choices(MAX_PHYSICAL_BATCH_SIZEs)[0]
        TARGET_EPOCHS = random.choices(TARGET_EPOCHSs)[0]

        EPSILON_index_list = [i for i, _ in enumerate(per_epoch_EPSILONs)]
        EPSILON_i = random.choices(EPSILON_index_list, weights=EPSILONs_weights)[0]
        EPSILON = per_epoch_EPSILONs[EPSILON_i]

        DELTA = 1e-8
        dispatcher_ip = DISPATCHER_IP
        dispatcher_port = DISPATCHER_PORT
        
        current_lambda = random.random() / 10
        last_arrival_time = poisson_arrival_times(last_arrival_time, current_lambda)
        job = generate_normal_one_job(
            last_arrival_time, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA, 
            update_sched_epoch_num, TARGET_EPOCHS, dispatcher_ip, dispatcher_port, is_history
        )
        jobs.append(job)

        current_decision_num += int(TARGET_EPOCHS / update_sched_epoch_num)
    return jobs
