import numpy as np
import random
import itertools
import json
import os
from utils.global_variable import DISPATCHER_IP, DISPATCHER_PORT, RECONSTRUCT_TRACE_PREFIX_PATH

waiting_select_model_names = ["CNN", "FF"]
waiting_select_train_dataset_names = ["EMNIST"]
waiting_select_test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]
waiting_select_datablock_select_num = [1, 2, 4]

waiting_select_products = list(itertools.product(
    waiting_select_model_names, waiting_select_train_dataset_names, waiting_select_test_dataset_names, waiting_select_datablock_select_num
))

def get_specific_model_config(model_name):
    if model_name == "CNN":
        return {
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "TARGET_EPOCHS": 50
        }
    elif model_name == "FF":
        return {
            "BATCH_SIZE": 1024,
            "MAX_PHYSICAL_BATCH_SIZE": 512,
            "TARGET_EPOCHS": 50
        }

def generate_dataset(dataset_names, fix_epsilon=10.0, fix_delta=1e-5, fix_time=0, num=6, save_path=""):
    print("check dataset_names: {}".format(dataset_names))
    result = {}
    for name in dataset_names:
        if name not in waiting_select_train_dataset_names:
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
    if len(save_path) > 0:
        dataset_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/datasets.json".format(save_path)
        if not os.path.exists(dataset_path):
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w+") as f:
            json.dump(result, f)
        print("save dataset trace in {}".format(dataset_path))
    return result

def generate_job_type(model_name, train_dataset_name, test_dataset_name, datablock_select_num):
    waiting_select_tuple = (model_name, train_dataset_name, test_dataset_name, datablock_select_num)
    index = waiting_select_products.index(waiting_select_tuple)
    job_type_identifier = "job_type_{}".format(index)
    return job_type_identifier

def generate_normal_one_job(time, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA, 
                            TARGET_EPOCHS, dispatcher_ip, dispatcher_port,
                            is_history=False):
    if (is_history and time > 0) or (not is_history and time < 0):
        time = -time
    job_type = generate_job_type(model_name, train_dataset_name, test_dataset_name, datablock_select_num)
    job_detail = {
        "time": time,
        "model_name": model_name,
        "job_type": job_type,
        "train_dataset_name": train_dataset_name,
        "test_dataset_name": test_dataset_name,
        "sub_test_key_id": "test_sub_0",
        "datablock_select_num": datablock_select_num,
        "LR": 1e-3,
        "EPSILON": EPSILON,
        "DELTA": DELTA,
        # "update_sched_epoch_num": update_sched_epoch_num,
        # "MAX_EPOCHS": TARGET_EPOCHS * 2,
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

def generate_jobs(all_decision_num, per_epoch_EPSILONs, EPSILONs_weights, time_interval, is_history, save_path=""):
    # 在这里应该生成比较多的类型
    # all_big_job_num = int(all_decision_num / update_sched_epoch_num)
    # lambdas = [random.random() / 10 for _ in range(all_big_job_num)]
    # all_big_job_arrival_times = poisson_arrival_times(all_big_job_num, lambdas)
    
    # 从一大堆里面生成
    models = ["CNN"]
    models_weights = [1.0]
    
    train_dataset_names = ["EMNIST"]
    
    test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]
    test_dataset_names_weights = [0.8, 0.15, 0.05]

    datablock_select_nums = [1, 2, 4]
    datablock_select_nums_weights = [0.6, 0.3, 0.1]
    
    jobs = []
    current_decision_num = 0
    last_arrival_time = 0.0
    while current_decision_num < all_decision_num:
        model_name_index_list = [i for i, _ in enumerate(models)]
        model_name_i = random.choices(model_name_index_list)[0]
        model_name = models[model_name_i]
        
        details = get_specific_model_config(model_name)
        BATCH_SIZE = details["BATCH_SIZE"]
        MAX_PHYSICAL_BATCH_SIZE = details["MAX_PHYSICAL_BATCH_SIZE"]
        TARGET_EPOCHS = details["TARGET_EPOCHS"]
        
        train_dataset_name = random.choices(train_dataset_names)[0]

        test_dataset_names_index_list = [i for i, _ in enumerate(test_dataset_names)]
        test_dataset_name_i = random.choices(test_dataset_names_index_list, weights=test_dataset_names_weights)[0]
        test_dataset_name = test_dataset_names[test_dataset_name_i]

        datablock_select_nums_index_list = [i for i, _ in enumerate(datablock_select_nums)]
        datablock_select_num_i = random.choices(datablock_select_nums_index_list, weights=datablock_select_nums_weights)[0]
        datablock_select_num = datablock_select_nums[datablock_select_num_i]


        EPSILON_index_list = [i for i, _ in enumerate(per_epoch_EPSILONs)]
        EPSILON_i = random.choices(EPSILON_index_list, weights=EPSILONs_weights)[0]
        EPSILON = per_epoch_EPSILONs[EPSILON_i]

        DELTA = 1e-8
        dispatcher_ip = DISPATCHER_IP
        dispatcher_port = DISPATCHER_PORT
        
        current_lambda = 1 / time_interval
        last_arrival_time = poisson_arrival_times(last_arrival_time, current_lambda)
        job = generate_normal_one_job(
            last_arrival_time, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA, 
            TARGET_EPOCHS, dispatcher_ip, dispatcher_port, is_history
        )
        jobs.append(job)

        current_decision_num += 1

    if len(save_path) > 0:
        if is_history:
            job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/his_jobs.json".format(save_path)
        else:
            job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/test_jobs.json".format(save_path)
        if not os.path.exists(job_path):
            os.makedirs(os.path.dirname(job_path), exist_ok=True)
        with open(job_path, "w+") as f:
            json.dump(jobs, f)
        
    return jobs
