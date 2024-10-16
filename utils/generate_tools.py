import numpy as np
import random
import itertools
import json
import os
import pandas as pd
from utils.global_variable import ALIBABA_DP_TRACE_PATH, RESULT_PATH
from utils.global_functions import NpEncoder

waiting_select_model_names = ["CNN", "FF"]
waiting_select_train_dataset_names = ["EMNIST"]
waiting_select_test_dataset_names = ["EMNIST-2000", "EMNIST_MNIST-1000_1000", "MNIST-2000"]

waiting_select_products = list(itertools.product(
    waiting_select_model_names, waiting_select_train_dataset_names, waiting_select_test_dataset_names
))

def generate_job_type(model_name, train_dataset_name, test_dataset_name):
    waiting_select_tuple = (model_name, train_dataset_name, test_dataset_name)
    index = waiting_select_products.index(waiting_select_tuple)
    job_type_identifier = "job_type_{}".format(index)
    return job_type_identifier

def change_dispatcher_ip_port(job_detail, dispatcher_ip, dispatcher_port):
    job_detail["dispatcher_ip"] = dispatcher_ip
    job_detail["dispatcher_port"] = dispatcher_port
    return job_detail

def change_arrival_time(detail, arrival_time):
    detail["time"] = arrival_time
    return detail

def change_epsilon(job_detail, new_epsilon):
    print("origin job_detail[EPSILON]: {} vs. new_epsilon: {}".format(job_detail["EPSILON"], new_epsilon))
    job_detail["EPSILON"] = new_epsilon
    return job_detail

def change_epsilon_G(datablock_detail, new_epsilon):
    print("origin datablock_detail[epsilon_capacity]: {} vs. new_epsilon: {}".format(datablock_detail["epsilon_capacity"], new_epsilon))
    datablock_detail["epsilon_capacity"] = new_epsilon
    return datablock_detail

def generate_normal_one_job(time, model_name, train_dataset_name, test_dataset_name, target_datablock_select_num, 
                            BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON_PER_EPOCH, DELTA, 
                            TARGET_EPOCHS, SITON_RUN_EPOCH_NUM, TAGRET_ACC, dispatcher_ip, dispatcher_port,
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
        "datablock_select_num": target_datablock_select_num,
        "LR": 1e-3,
        "EPSILON": EPSILON_PER_EPOCH,
        "DELTA": DELTA,
        "TARGET_EPOCHS": TARGET_EPOCHS,
        "SITON_RUN_EPOCH_NUM": SITON_RUN_EPOCH_NUM,
        "MAX_GRAD_NORM": 1.2,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_PHYSICAL_BATCH_SIZE": MAX_PHYSICAL_BATCH_SIZE,
        "TAGRET_ACC": TAGRET_ACC,
        "simulation_init_test_acc": 0.2,
        "siton_up_test_acc": 0.05,
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

def generate_alibaba_jobs(all_num, offline_num,
                time_speed_up, is_history,
                valid_max_epsilon_require, valid_min_epsilon_require,
                job_require_select_block_min_num, job_require_select_block_max_num,
                change_job_epsilon_max_times,
                dispatcher_ip, dispatcher_port, enable_waiting_flag,
                jobtrace_reconstruct_path="", save_path=""):
    jobs = []
    offline_time_default = 0.0
    online_time_iterval = 12.0 / time_speed_up # 1分钟1个块?

    print("check valid_min_epsilon_require: {}".format(valid_min_epsilon_require))
    print("check max_job_epsilon_require: {}".format(valid_max_epsilon_require))

    if len(jobtrace_reconstruct_path) > 0:
        if is_history:
            his_job_path = RESULT_PATH + "/{}/his_jobs.json".format(jobtrace_reconstruct_path)
            with open(his_job_path, "r+") as f:
                temp_jobs = json.load(f)
        else:
            test_job_path = RESULT_PATH + "/{}/test_jobs.json".format(jobtrace_reconstruct_path)
            with open(test_job_path, "r+") as f:
                temp_jobs = json.load(f)
        
        current_decision_num = 0
        current_search_pointer_jobid = 0
        while current_decision_num < all_num and current_search_pointer_jobid < len(temp_jobs):
            job_detail = temp_jobs[current_search_pointer_jobid]
            
            check_flag = True
            if valid_min_epsilon_require is not None and job_detail["EPSILON"] * job_detail["SITON_RUN_EPOCH_NUM"] < valid_min_epsilon_require:
                check_flag = False
            if valid_max_epsilon_require is not None and job_detail["EPSILON"] * job_detail["SITON_RUN_EPOCH_NUM"] > valid_max_epsilon_require:
                check_flag = False
            if job_require_select_block_min_num is not None and job_detail["datablock_select_num"] < job_require_select_block_min_num:
                check_flag = False
            if job_require_select_block_max_num is not None and job_detail["datablock_select_num"] > job_require_select_block_max_num:
                check_flag = False
            if check_flag:
                temp_jobs[current_search_pointer_jobid] = change_dispatcher_ip_port(job_detail, dispatcher_ip, dispatcher_port)
                if change_job_epsilon_max_times != 1.0:
                    temp_jobs[current_search_pointer_jobid] = change_epsilon(job_detail, job_detail["EPSILON"] * change_job_epsilon_max_times)
                jobs.append(temp_jobs[current_search_pointer_jobid])
                current_decision_num += 1
            current_search_pointer_jobid += 1
    else:
        alibaba_dp_trace_path = ALIBABA_DP_TRACE_PATH + "/privacy_tasks_30_days_extend.csv"
        valid_sample_df = pd.read_csv(alibaba_dp_trace_path)

        if valid_min_epsilon_require is not None:
            valid_sample_df = valid_sample_df[valid_sample_df["epsilon_per_epoch"] * valid_sample_df["siton_run_epoch_num"] >= valid_min_epsilon_require]
            print("check valid_min_epsilon_require: {}".format(valid_min_epsilon_require))
            print("check valid_sample_df: {}".format(len(valid_sample_df)))
        if valid_max_epsilon_require is not None:
            valid_sample_df = valid_sample_df[valid_sample_df["epsilon_per_epoch"] * valid_sample_df["siton_run_epoch_num"] <= valid_max_epsilon_require]
            print("check valid_max_epsilon_require: {}".format(valid_max_epsilon_require))
            print("check valid_sample_df: {}".format(len(valid_sample_df)))
        
        print("check valid_sample_df: {}".format(len(valid_sample_df)))
        if job_require_select_block_min_num is not None:
            valid_sample_df = valid_sample_df[valid_sample_df["n_blocks"] >= job_require_select_block_min_num]
            print("check job_require_select_block_min_num: {}".format(job_require_select_block_min_num))
            print("check valid_sample_df: {}".format(len(valid_sample_df)))
        if job_require_select_block_max_num is not None:
            valid_sample_df = valid_sample_df[valid_sample_df["n_blocks"] <= job_require_select_block_max_num]
            print("check job_require_select_block_max_num: {}".format(job_require_select_block_max_num))
            print("check valid_sample_df: {}".format(len(valid_sample_df)))
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
        result_df = result_df.sort_values("n_blocks")
        print("check result_df len: {}".format(len(result_df)))

        current_decision_num = 0
        while current_decision_num < all_num:
            result_df_line = result_df.iloc[current_decision_num]
            model_name = result_df_line["model"]
            
            BATCH_SIZE = result_df_line["batch_size"]
            MAX_PHYSICAL_BATCH_SIZE = result_df_line["max_physical_batch_size"]
            TARGET_EPOCHS = result_df_line["target_epochs"]
            SITON_RUN_EPOCH_NUM = result_df_line["siton_run_epoch_num"]
            TAGRET_ACC = result_df_line["target_acc"]
            
            train_dataset_name = result_df_line["train_dataset_name"]
            test_dataset_name = result_df_line["test_dataset_name"]

            datablock_select_num = result_df_line["n_blocks"]
            EPSILON_PER_EPOCH = result_df_line["epsilon_per_epoch"]

            DELTA = result_df_line["delta"]

            arrival_time_zhanwei = offline_time_default
            job = generate_normal_one_job(
                arrival_time_zhanwei, model_name, train_dataset_name, test_dataset_name, datablock_select_num, 
                BATCH_SIZE, MAX_PHYSICAL_BATCH_SIZE, EPSILON_PER_EPOCH, DELTA, 
                TARGET_EPOCHS, SITON_RUN_EPOCH_NUM, TAGRET_ACC, dispatcher_ip, dispatcher_port, 
                is_history
            )

            if change_job_epsilon_max_times != 1.0:
                job = change_epsilon(job, job["EPSILON"] * change_job_epsilon_max_times)
            jobs.append(job)
            current_decision_num += 1
        print("current_decision_num: {}".format(current_decision_num))

    jobs = sorted(jobs, key=lambda x: x["time"])
    # 各类归一化操作: 时间归一化
    if len(jobs) > 0:
        all_norm_time = []
        all_time_interval = []
        last_time = 0.0
        for job_detail_index, job_detail in enumerate(jobs):
            if enable_waiting_flag:
                norm_time = offline_time_default
            elif job_detail_index < offline_num:
                norm_time = offline_time_default
            else:
                norm_time = online_time_iterval * (job_detail_index - offline_num)
            jobs[job_detail_index] = change_arrival_time(job_detail, norm_time)
            all_norm_time.append(norm_time)
            all_time_interval.append(norm_time - last_time)
            last_time = norm_time
        print(f"norm_time => min: {np.min(all_norm_time)}; max: {np.max(all_norm_time)}")
        print(f"all_time_interval => min: {np.min(all_time_interval)}; max: {np.max(all_time_interval)}; mean: {np.mean(all_time_interval)}")

    if len(save_path) > 0:
        if is_history:
            job_path = RESULT_PATH + "/{}/his_jobs.json".format(save_path)
        else:
            job_path = RESULT_PATH + "/{}/test_jobs.json".format(save_path)
        if not os.path.exists(job_path):
            os.makedirs(os.path.dirname(job_path), exist_ok=True)
        with open(job_path, "w+") as f: # 覆盖写
            json.dump(jobs, f, cls=NpEncoder)
    return jobs

def generate_alibaba_dataset(num, offline_num, time_speed_up,
                    dataset_names, fix_epsilon, fix_delta,
                    dataset_reconstruct_path="", save_path=""):
    offline_time_default = -100.0
    online_time_iterval = 120.0 / time_speed_up # 1分钟1个块?
    datasets_list = {}
    time_2_datablock_num = {}
    if len(dataset_reconstruct_path) > 0:
        print("load from path: {}".format(dataset_reconstruct_path))
        dataset_path = RESULT_PATH + "/{}/datasets.json".format(dataset_reconstruct_path)
        with open(dataset_path, "r+") as f:
            temp_datasets_list = json.load(f)
        current_num = 0
        for name in temp_datasets_list:
            if name not in datasets_list:
                datasets_list[name] = {}
            for sub_datablock_name in temp_datasets_list[name]:
                datasets_list[name][sub_datablock_name] = temp_datasets_list[name][sub_datablock_name]
                if datasets_list[name][sub_datablock_name]["epsilon_capacity"] != fix_epsilon:
                    datasets_list[name][sub_datablock_name] = change_epsilon_G(datasets_list[name][sub_datablock_name], fix_epsilon)
                current_num += 1
                if current_num >= num:
                    break
            if current_num >= num:
                break
    else:
        print("check dataset_names: {}".format(dataset_names))
        for name in dataset_names:
            if name not in waiting_select_train_dataset_names:
                continue
            datasets_list[name] = {}
            for index in range(num):
                sub_datablock_name = "train_sub_{}".format(index)
                arrival_time_zhanwei = offline_time_default
                datasets_list[name][sub_datablock_name] = {
                    "submited": False,
                    "epsilon_capacity": fix_epsilon,
                    "delta_capacity": fix_delta,
                    "time": arrival_time_zhanwei
                }
            print(datasets_list[name])
    
    # 根据时间排序, 然后顺序延迟赋值时间, 这样就实现了动态更新!
    time_index = 0
    for name in datasets_list:
        for sub_datablock_name in datasets_list[name]:
            if time_index < offline_num:
                arrival_time = offline_time_default
            else:
                arrival_time = online_time_iterval * (time_index - offline_num)
            time_index += 1
            datasets_list[name][sub_datablock_name] = change_arrival_time(datasets_list[name][sub_datablock_name], arrival_time)
            if arrival_time not in time_2_datablock_num:
                time_2_datablock_num[arrival_time] = 0
            time_2_datablock_num[arrival_time] += 1
    
    if len(save_path) > 0:
        dataset_path = RESULT_PATH + "/{}/datasets.json".format(save_path)
        if not os.path.exists(dataset_path):
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w+") as f: # 覆盖写
            json.dump(datasets_list, f)
        print("save dataset trace in {}".format(dataset_path))
    return datasets_list, time_2_datablock_num