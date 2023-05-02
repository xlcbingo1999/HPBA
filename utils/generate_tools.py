import numpy as np
import random
import itertools
import json
import os
import pandas as pd
from utils.global_variable import RECONSTRUCT_TRACE_PREFIX_PATH, ALIBABA_DP_TRACE_PATH
from utils.global_functions import NpEncoder

waiting_select_model_names = ["CNN", "FF"]
waiting_select_train_dataset_names = ["EMNIST"]
waiting_select_test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]

waiting_select_products = list(itertools.product(
    waiting_select_model_names, waiting_select_train_dataset_names, waiting_select_test_dataset_names
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

def generate_dataset(dataset_names, fix_epsilon=10.0, fix_delta=1e-5, fix_time=0, num=6, dataset_reconstruct_path="", save_path=""):
    if len(dataset_reconstruct_path) > 0:
        print("load from path: {}".format(dataset_reconstruct_path))
        dataset_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/datasets.json".format(dataset_reconstruct_path)
        with open(dataset_path, "r+") as f:
            datasets_list = json.load(f)
    else:
        print("check dataset_names: {}".format(dataset_names))
        datasets_list = {}
        for name in dataset_names:
            if name not in waiting_select_train_dataset_names:
                continue
            datasets_list[name] = {}
            for index in range(num):
                sub_datablock_name = "train_sub_{}".format(index)
                datasets_list[name][sub_datablock_name] = {
                    "submited": False,
                    "epsilon_capacity": fix_epsilon,
                    "delta_capacity": fix_delta,
                    "time": fix_time
                }
            print(datasets_list[name])
    if len(save_path) > 0:
        dataset_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/datasets.json".format(save_path)
        if not os.path.exists(dataset_path):
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w+") as f:
            json.dump(datasets_list, f)
        print("save dataset trace in {}".format(dataset_path))
    return datasets_list

def generate_job_type(model_name, train_dataset_name, test_dataset_name):
    waiting_select_tuple = (model_name, train_dataset_name, test_dataset_name)
    index = waiting_select_products.index(waiting_select_tuple)
    job_type_identifier = "job_type_{}".format(index)
    return job_type_identifier

def change_dispatcher_ip_port(job_detail, dispatcher_ip, dispatcher_port):
    job_detail["dispatcher_ip"] = dispatcher_ip
    job_detail["dispatcher_port"] = dispatcher_port
    return job_detail

def change_arrival_time(job_detail, arrival_time):
    job_detail["time"] = arrival_time
    return job_detail

def generate_normal_one_job(time, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA, 
                            TARGET_EPOCHS, dispatcher_ip, dispatcher_port,
                            is_history=False):
    if (is_history and time > 0) or (not is_history and time < 0):
        time = -time
    job_type = generate_job_type(model_name, train_dataset_name, test_dataset_name)
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

def generate_jobs(all_num, per_epoch_EPSILONs, EPSILONs_weights, 
                time_interval, need_change_interval, is_history, 
                dispatcher_ip, dispatcher_port,
                jobtrace_reconstruct_path="", save_path=""):
    if len(jobtrace_reconstruct_path) > 0:
        if is_history:
            his_job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/his_jobs.json".format(jobtrace_reconstruct_path)
            with open(his_job_path, "r+") as f:
                jobs = json.load(f)
        else:
            test_job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/test_jobs.json".format(jobtrace_reconstruct_path)
            with open(test_job_path, "r+") as f:
                jobs = json.load(f)
        current_decision_num = 0
        last_arrival_time = 0.0
        for job_detail in jobs:
            job_detail = change_dispatcher_ip_port(job_detail, dispatcher_ip, dispatcher_port)
            if need_change_interval:
                if current_decision_num > 0:
                    current_lambda = 1 / time_interval
                    last_arrival_time = poisson_arrival_times(last_arrival_time, current_lambda)
                job_detail = change_arrival_time(job_detail, last_arrival_time)
                current_decision_num += 1
    else:
        # 从一大堆里面生成
        models = ["CNN", "FF"]
        models_weights = [0.5, 0.5]
        
        train_dataset_names = ["EMNIST"]
        
        test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]
        test_dataset_names_weights = [0.8, 0.15, 0.05]

        datablock_select_nums = [1, 2, 4]
        datablock_select_nums_weights = [0.6, 0.3, 0.1]
        
        jobs = []
        current_decision_num = 0
        last_arrival_time = 0.0
        while current_decision_num < all_num:
            model_name_index_list = [i for i, _ in enumerate(models)]
            model_name_i = random.choices(model_name_index_list, weights=models_weights)[0]
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

            if current_decision_num > 0:
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

def generate_alibaba_jobs(all_num, 
                time_speed_up, need_change_interval, is_history,
                datablock_require_epsilon_max_ratio, min_epsilon_capacity,
                dispatcher_ip, dispatcher_port,
                jobtrace_reconstruct_path="", save_path=""):
    if len(jobtrace_reconstruct_path) > 0:
        if is_history:
            his_job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/his_jobs.json".format(jobtrace_reconstruct_path)
            with open(his_job_path, "r+") as f:
                jobs = json.load(f)
        else:
            test_job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/test_jobs.json".format(jobtrace_reconstruct_path)
            with open(test_job_path, "r+") as f:
                jobs = json.load(f)
        current_decision_num = 0
        for job_detail in jobs:
            job_detail = change_dispatcher_ip_port(job_detail, dispatcher_ip, dispatcher_port)
            if need_change_interval:
                old_time = job_detail["time"]
                new_arrival_time = old_time / time_speed_up
                job_detail = change_arrival_time(job_detail, new_arrival_time)
                current_decision_num += 1
    else:
        alibaba_dp_trace_path = ALIBABA_DP_TRACE_PATH + "/privacy_tasks_30_days.csv"
        df = pd.read_csv(alibaba_dp_trace_path)
        if datablock_require_epsilon_max_ratio is not None:
            max_job_epsilon_require = min_epsilon_capacity * datablock_require_epsilon_max_ratio
            valid_sample_df = df[df["epsilon"] < max_job_epsilon_require]
            print("min df[epsilon]: {}".format(min(df["epsilon"])))
            print("check max_job_epsilon_require: {}".format(max_job_epsilon_require))
            print("check valid_sample_df: {}".format(len(valid_sample_df)))
        else:
            valid_sample_df = df
        if len(valid_sample_df) > all_num:
            result_df = valid_sample_df.sample(n=all_num, replace=False)
        else:
            result_df = valid_sample_df.copy()
            while len(result_df) < all_num:
                if len(valid_sample_df) > all_num - len(result_df):
                    temp_df = valid_sample_df.sample(n=all_num - len(result_df), replace=False)
                else:
                    temp_df = valid_sample_df.copy()
                result_df = pd.concat([result_df, temp_df])
                # print("check result_df: len {}".format(len(result_df)))
        print("check result_df len: {}".format(len(result_df)))

        # 从一大堆里面生成
        models = ["CNN", "FF"]
        models_weights = [0.5, 0.5]
        
        train_dataset_names = ["EMNIST"]
        
        test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]
        test_dataset_names_weights = [0.8, 0.15, 0.05]
        
        jobs = []
        current_decision_num = 0
        last_arrival_time = 0.0
        while current_decision_num < all_num:
            result_df_line = result_df.iloc[current_decision_num]
            model_name_index_list = [i for i, _ in enumerate(models)]
            model_name_i = random.choices(model_name_index_list, weights=models_weights)[0]
            model_name = models[model_name_i]
            
            details = get_specific_model_config(model_name)
            BATCH_SIZE = details["BATCH_SIZE"]
            MAX_PHYSICAL_BATCH_SIZE = details["MAX_PHYSICAL_BATCH_SIZE"]
            TARGET_EPOCHS = details["TARGET_EPOCHS"]
            
            train_dataset_name = random.choices(train_dataset_names)[0]

            test_dataset_names_index_list = [i for i, _ in enumerate(test_dataset_names)]
            test_dataset_name_i = random.choices(test_dataset_names_index_list, weights=test_dataset_names_weights)[0]
            test_dataset_name = test_dataset_names[test_dataset_name_i]

            datablock_select_num = result_df_line["n_blocks"]
            epsilon_all_epochs = result_df_line["epsilon"]

            EPSILON = epsilon_all_epochs / TARGET_EPOCHS
            DELTA = result_df_line["delta"]

            if need_change_interval:
                arrival_time = result_df_line["norm_submit_time"] / time_speed_up
            else:
                arrival_time = result_df_line["norm_submit_time"]
            job = generate_normal_one_job(
                arrival_time, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
                BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON, DELTA, 
                TARGET_EPOCHS, dispatcher_ip, dispatcher_port, is_history
            )
            jobs.append(job)

            current_decision_num += 1
        print("current_decision_num: {}".format(current_decision_num))

    if len(save_path) > 0:
        if is_history:
            job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/his_jobs.json".format(save_path)
        else:
            job_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/test_jobs.json".format(save_path)
        if not os.path.exists(job_path):
            os.makedirs(os.path.dirname(job_path), exist_ok=True)
        with open(job_path, "w+") as f:
            json.dump(jobs, f, cls=NpEncoder)
    return jobs

def generate_alibaba_dataset(num, offline_num, time_speed_up,
                    dataset_names, fix_epsilon, fix_delta,
                    dataset_reconstruct_path="", save_path=""):
    if len(dataset_reconstruct_path) > 0:
        print("load from path: {}".format(dataset_reconstruct_path))
        dataset_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/datasets.json".format(dataset_reconstruct_path)
        with open(dataset_path, "r+") as f:
            datasets_list = json.load(f)
    else:
        print("check dataset_names: {}".format(dataset_names))
        offline_num = offline_num
        online_time_iterval = 3600.0 * 4 / time_speed_up
        datasets_list = {}
        for name in dataset_names:
            if name not in waiting_select_train_dataset_names:
                continue
            datasets_list[name] = {}
            for index in range(num):
                sub_datablock_name = "train_sub_{}".format(index)
                if index < offline_num:
                    arrival_time = -100.0
                else:
                    arrival_time = online_time_iterval * (index - offline_num)
                datasets_list[name][sub_datablock_name] = {
                    "submited": False,
                    "epsilon_capacity": fix_epsilon,
                    "delta_capacity": fix_delta,
                    "time": arrival_time
                }
            print(datasets_list[name])
    if len(save_path) > 0:
        dataset_path = RECONSTRUCT_TRACE_PREFIX_PATH + "/{}/datasets.json".format(save_path)
        if not os.path.exists(dataset_path):
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w+") as f:
            json.dump(datasets_list, f)
        print("save dataset trace in {}".format(dataset_path))
    return datasets_list