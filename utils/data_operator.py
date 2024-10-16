import os
import re
import ast
import numpy as np
import pandas as pd
from utils.global_variable import RESULT_PATH
from utils.global_functions import print_console_file

def is_number(s):    
    try:    # 如果能运⾏ float(s) 语句，返回 True（字符串 s 是浮点数）        
        float(s)        
        return True    
    except ValueError:  # ValueError 为 Python 的⼀种标准异常，表⽰"传⼊⽆效的参数"        
        pass  # 如果引发了 ValueError 这种异常，不做任何事情（pass：不做任何事情，⼀般⽤做占位语句）    
    try:        
        import unicodedata  # 处理 ASCII 码的包        
        unicodedata.numeric(s)  # 把⼀个表⽰数字的字符串转换为浮点数返回的函数        
        return True    
    except (TypeError, ValueError):        
        pass    
        return False

def log_args_var(args, all_result_path):
    args_message = '\n'.join([f'{k}: {v}' for k, v in vars(args).items()])
    with open(all_result_path, "a+") as f:
        print_console_file("===== args_message =====", fileHandler=f)
        print_console_file(args_message, fileHandler=f)

def final_operate_data_old_version(current_test_all_dir):
    print(f"=============================== {current_test_all_dir} =================================")
    trace_save_path = os.path.join(RESULT_PATH, current_test_all_dir)
    final_used_num, success_num_arr, failed_num_arr, all_final_significance_arr, success_final_significance_arr, \
        success_datablock_num_arr, failed_datablock_num_arr, target_datablock_num_arr = read_DL_dispatcher_result_func_simulation_old_version(trace_save_path)
    
    # 新建一个全新的log进行保存
    all_result_path = "{}/all_result_backward.log".format(trace_save_path)
    
    with open(all_result_path, "a+") as f:
        print_console_file("final_used_num: {}".format(final_used_num), fileHandler=f)
        if final_used_num > 0:
            print_console_file("[success_info] {}({}~{}) === success_num_mean: {} ; success_num_min: {} ; success_num_max: {}".format(
                np.mean(success_num_arr), min(success_num_arr), max(success_num_arr), np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
            ), fileHandler=f)
            print_console_file("[failed_info] {}({}~{}) === failed_num_mean: {} ; failed_num_min: {} ; failed_num_max: {}".format(
                np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr), np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
            ), fileHandler=f)
            print_console_file("[allsig_info] {}({}~{}) === all_final_significance_mean: {} ; all_final_significance_min: {} ; all_final_significance_max: {}".format(
                np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr), np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
            ), fileHandler=f)
            print_console_file("[successsig_info] {}({}~{}) === success_final_significance_mean: {} ; success_final_significance_min: {} ; success_final_significance_max: {}".format(
                np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr), np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr)
            ), fileHandler=f)
            print_console_file("[successblock_info] {}({}~{}) === success_datablock_num_mean: {} ; success_datablock_num_min: {} ; success_datablock_num_max: {}".format(
                np.mean(success_datablock_num_arr), min(success_datablock_num_arr), max(success_datablock_num_arr), np.mean(success_datablock_num_arr), min(success_datablock_num_arr), max(success_datablock_num_arr)
            ), fileHandler=f)
            print_console_file("[failedblock_info] {}({}~{}) === failed_datablock_num_mean: {} ; failed_datablock_num_min: {} ; failed_datablock_num_max: {}".format(
                np.mean(failed_datablock_num_arr), min(failed_datablock_num_arr), max(failed_datablock_num_arr), np.mean(failed_datablock_num_arr), min(failed_datablock_num_arr), max(failed_datablock_num_arr)
            ), fileHandler=f)
            print_console_file("[targetblock_info] {}({}~{}) === target_datablock_num_mean: {} ; target_datablock_num_min: {} ; target_datablock_num_max: {}".format(
                np.mean(target_datablock_num_arr), min(target_datablock_num_arr), max(target_datablock_num_arr), np.mean(target_datablock_num_arr), min(target_datablock_num_arr), max(target_datablock_num_arr)
            ), fileHandler=f)
    print(f"============================ {current_test_all_dir} [success:{final_used_num}] ====================================")
    return final_used_num, success_num_arr, failed_num_arr, all_final_significance_arr, success_final_significance_arr, \
        success_datablock_num_arr, failed_datablock_num_arr, target_datablock_num_arr

def final_log_result(current_test_all_dir, all_result_file_name):
    trace_save_all_file_dir = os.path.join(RESULT_PATH, current_test_all_dir)
    all_result_path = os.path.join(trace_save_all_file_dir, all_result_file_name)

    final_used_num, success_num_arr, failed_num_arr, all_test_jobs_num_arr, \
        all_train_loss_arr, all_train_accuracy_arr, \
        all_test_loss_arr, all_test_accuracy_arr, all_final_significance_arr, \
        all_target_datablock_num_arr, all_success_datablock_num_arr, all_failed_datablock_num_arr, \
        all_epsilon_real_all_block_arr, all_significance_2_epsilon_real_blocks_arr, \
        all_test_acc_2_epsilon_real_blocks_arr, all_test_loss_2_epsilon_real_blocks_arr, \
        all_decision_duration_arr = read_DL_dispatcher_result_func(trace_save_all_file_dir)

    with open(all_result_path, "w+") as f:
        print_console_file("final_used_num: {}".format(final_used_num), fileHandler=f)
        if final_used_num > 0:
            print_console_file("[success_info] {}({}~{}) === success_num_mean: {} ; success_num_min: {} ; success_num_max: {}".format(
                np.mean(success_num_arr), min(success_num_arr), max(success_num_arr), np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
            ), fileHandler=f)
            print_console_file("[failed_info] {}({}~{}) === failed_num_mean: {} ; failed_num_min: {} ; failed_num_max: {}".format(
                np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr), np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
            ), fileHandler=f)
            print_console_file("[testjob_info] {}({}~{}) === testjob_num_mean: {} ; testjob_num_min: {} ; testjob_num_max: {}".format(
                np.mean(all_test_jobs_num_arr), min(all_test_jobs_num_arr), max(all_test_jobs_num_arr), np.mean(all_test_jobs_num_arr), min(all_test_jobs_num_arr), max(all_test_jobs_num_arr)
            ), fileHandler=f)

            print_console_file("[allsig_info] {}({}~{}) === all_final_significance_mean: {} ; all_final_significance_min: {} ; all_final_significance_max: {}".format(
                np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr), np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
            ), fileHandler=f)
            print_console_file("[successblock_info] {}({}~{}) === success_datablock_num_mean: {} ; success_datablock_num_min: {} ; success_datablock_num_max: {}".format(
                np.mean(all_success_datablock_num_arr), min(all_success_datablock_num_arr), max(all_success_datablock_num_arr), np.mean(all_success_datablock_num_arr), min(all_success_datablock_num_arr), max(all_success_datablock_num_arr)
            ), fileHandler=f)
            print_console_file("[failedblock_info] {}({}~{}) === failed_datablock_num_mean: {} ; failed_datablock_num_min: {} ; failed_datablock_num_max: {}".format(
                np.mean(all_failed_datablock_num_arr), min(all_failed_datablock_num_arr), max(all_failed_datablock_num_arr), np.mean(all_failed_datablock_num_arr), min(all_failed_datablock_num_arr), max(all_failed_datablock_num_arr)
            ), fileHandler=f)
            print_console_file("[targetblock_info] {}({}~{}) === target_datablock_num_mean: {} ; target_datablock_num_min: {} ; target_datablock_num_max: {}".format(
                np.mean(all_target_datablock_num_arr), min(all_target_datablock_num_arr), max(all_target_datablock_num_arr), np.mean(all_target_datablock_num_arr), min(all_target_datablock_num_arr), max(all_target_datablock_num_arr)
            ), fileHandler=f)

            print_console_file("[train_loss_info] {}({}~{}) === train_loss_mean: {} ; train_loss_min: {} ; train_loss_max: {}".format(
                np.mean(all_train_loss_arr), min(all_train_loss_arr), max(all_train_loss_arr), np.mean(all_train_loss_arr), min(all_train_loss_arr), max(all_train_loss_arr)
            ), fileHandler=f)
            print_console_file("[train_accuracy_info] {}({}~{}) === train_accuracy_mean: {} ; train_accuracy_min: {} ; train_accuracy_max: {}".format(
                np.mean(all_train_accuracy_arr), min(all_train_accuracy_arr), max(all_train_accuracy_arr), np.mean(all_train_accuracy_arr), min(all_train_accuracy_arr), max(all_train_accuracy_arr)
            ), fileHandler=f)
            print_console_file("[test_loss_info] {}({}~{}) === test_loss_mean: {} ; test_loss_min: {} ; test_loss_max: {}".format(
                np.mean(all_test_loss_arr), min(all_test_loss_arr), max(all_test_loss_arr), np.mean(all_test_loss_arr), min(all_test_loss_arr), max(all_test_loss_arr)
            ), fileHandler=f)
            print_console_file("[test_accuracy_info] {}({}~{}) === test_accuracy_mean: {} ; test_accuracy_min: {} ; test_accuracy_max: {}".format(
                np.mean(all_test_accuracy_arr), min(all_test_accuracy_arr), max(all_test_accuracy_arr), np.mean(all_test_accuracy_arr), min(all_test_accuracy_arr), max(all_test_accuracy_arr)
            ), fileHandler=f)

            print_console_file("[epsilon_real_all_block_info] {}({}~{}) === epsilon_real_all_block_mean: {} ; epsilon_real_all_block_min: {} ; epsilon_real_all_block_max: {}".format(
                np.mean(all_epsilon_real_all_block_arr), min(all_epsilon_real_all_block_arr), max(all_epsilon_real_all_block_arr), np.mean(all_epsilon_real_all_block_arr), min(all_epsilon_real_all_block_arr), max(all_epsilon_real_all_block_arr)
            ), fileHandler=f)
            print_console_file("[significance_2_epsilon_real_blocks_info] {}({}~{}) === significance_2_epsilon_real_blocks_mean: {} ; significance_2_epsilon_real_blocks_min: {} ; significance_2_epsilon_real_blocks_max: {}".format(
                np.mean(all_significance_2_epsilon_real_blocks_arr), min(all_significance_2_epsilon_real_blocks_arr), max(all_significance_2_epsilon_real_blocks_arr), np.mean(all_significance_2_epsilon_real_blocks_arr), min(all_significance_2_epsilon_real_blocks_arr), max(all_significance_2_epsilon_real_blocks_arr)
            ), fileHandler=f)
            print_console_file("[test_loss_2_epsilon_real_blocks_info] {}({}~{}) === test_loss_2_epsilon_real_blocks_mean: {} ; test_loss_2_epsilon_real_blocks_min: {} ; test_loss_2_epsilon_real_blocks_max: {}".format(
                np.mean(all_test_loss_2_epsilon_real_blocks_arr), min(all_test_loss_2_epsilon_real_blocks_arr), max(all_test_loss_2_epsilon_real_blocks_arr), np.mean(all_test_loss_2_epsilon_real_blocks_arr), min(all_test_loss_2_epsilon_real_blocks_arr), max(all_test_loss_2_epsilon_real_blocks_arr)
            ), fileHandler=f)
            print_console_file("[test_acc_2_epsilon_real_blocks_info] {}({}~{}) === test_acc_2_epsilon_real_blocks_mean: {} ; test_acc_2_epsilon_real_blocks_min: {} ; test_acc_2_epsilon_real_blocks_max: {}".format(
                np.mean(all_test_acc_2_epsilon_real_blocks_arr), min(all_test_acc_2_epsilon_real_blocks_arr), max(all_test_acc_2_epsilon_real_blocks_arr), np.mean(all_test_acc_2_epsilon_real_blocks_arr), min(all_test_acc_2_epsilon_real_blocks_arr), max(all_test_acc_2_epsilon_real_blocks_arr)
            ), fileHandler=f)
            print_console_file("[decision_duration_info] {}({}~{}) === decision_duration_mean: {} ; decision_duration_min: {} ; decision_duration_max: {}".format(
                np.mean(all_decision_duration_arr), min(all_decision_duration_arr), max(all_decision_duration_arr), np.mean(all_decision_duration_arr), min(all_decision_duration_arr), max(all_decision_duration_arr)
            ), fileHandler=f)
            
    print(f"============================ {current_test_all_dir} [success:{final_used_num}] ====================================")
    return final_used_num, success_num_arr, failed_num_arr, all_test_jobs_num_arr, all_train_loss_arr, all_train_accuracy_arr, \
        all_test_loss_arr, all_test_accuracy_arr, all_final_significance_arr, \
        all_target_datablock_num_arr, all_success_datablock_num_arr, all_failed_datablock_num_arr, \
        all_epsilon_real_all_block_arr, all_significance_2_epsilon_real_blocks_arr, \
        all_test_loss_2_epsilon_real_blocks_arr, all_test_acc_2_epsilon_real_blocks_arr, \
        all_decision_duration_arr

def read_DL_dispatcher_result_func(trace_save_all_file_dir):
    final_used_num = 0
    all_need_iter_paths = []
    for file_dir in os.listdir(trace_save_all_file_dir):
        if "time_log" in file_dir:
            result_dispatcher_file_dir = os.path.join(trace_save_all_file_dir, file_dir)
            all_need_iter_paths.append(result_dispatcher_file_dir)
            final_used_num += 1

    success_num_arr = []
    failed_num_arr = []
    all_test_jobs_num_arr = []

    all_train_loss_arr = []
    all_train_accuracy_arr = []
    all_test_loss_arr = []
    all_test_accuracy_arr = []
    all_final_significance_arr = []
    all_target_datablock_num_arr = []
    all_success_datablock_num_arr = []
    all_failed_datablock_num_arr = []
    
    all_epsilon_real_all_block_arr = []
    all_significance_2_epsilon_real_blocks_arr = []
    all_test_acc_2_epsilon_real_blocks_arr = []
    all_test_loss_2_epsilon_real_blocks_arr = []

    all_decision_duration_arr = []

    for file_path in all_need_iter_paths:
        result_df = pd.read_csv(file_path)
        
        all_test_job_num = len(result_df)
        success_num_arr.append(len(result_df[result_df["success_flag"] == 1]))
        failed_num_arr.append(len(result_df[result_df["success_flag"] == 0]))
        all_test_jobs_num_arr.append(all_test_job_num)
        
        all_train_loss_arr.append(np.sum(list(result_df["train_loss"])))
        all_train_accuracy_arr.append(np.sum(list(result_df["train_acc"])))
        test_loss = np.sum(list(result_df["test_loss"]))
        all_test_loss_arr.append(test_loss)
        test_acc = np.sum(list(result_df["test_acc"]))
        all_test_accuracy_arr.append(test_acc)
        sig = np.sum(list(result_df["significance"]))
        all_final_significance_arr.append(sig)
        
        all_target_datablock_num_arr.append(np.sum(list(result_df["target_datablock_num"])))
        all_success_datablock_num_arr.append(np.sum(list(result_df["success_datablock_num"])))
        all_failed_datablock_num_arr.append(np.sum(list(result_df["target_datablock_num"] - result_df["success_datablock_num"])))


        epsilon_real = np.sum(list(result_df["epsilon_real_all_block"]))
        all_epsilon_real_all_block_arr.append(epsilon_real)
        
        all_significance_2_epsilon_real_blocks_arr.append(sig/epsilon_real)
        all_test_acc_2_epsilon_real_blocks_arr.append(test_acc/epsilon_real)
        all_test_loss_2_epsilon_real_blocks_arr.append(test_loss/epsilon_real)

        decision_duration = np.mean(list(result_df["decision_duration"]))
        all_decision_duration_arr.append(decision_duration)
        
    return final_used_num, success_num_arr, failed_num_arr, all_test_jobs_num_arr, \
            all_train_loss_arr, all_train_accuracy_arr, \
            all_test_loss_arr, all_test_accuracy_arr, all_final_significance_arr, \
            all_target_datablock_num_arr, all_success_datablock_num_arr, all_failed_datablock_num_arr, \
            all_epsilon_real_all_block_arr, all_significance_2_epsilon_real_blocks_arr, \
            all_test_acc_2_epsilon_real_blocks_arr, all_test_loss_2_epsilon_real_blocks_arr, \
            all_decision_duration_arr

        

def read_DL_dispatcher_result_func_zhengze(trace_save_path):
    success_fail_num_pattern = r'current_success_num:\s*(?P<success>[-+]?\d+(?:\.\d+)?);\s+current_failed_num:\s*(?P<failed>[-+]?\d+(?:\.\d+)?);\s+current_no_submit_num:\s*(?P<no_submit>[-+]?\d+(?:\.\d+)?);\s+current_no_sche_num:\s*(?P<no_sche>[-+]?\d+(?:\.\d+)?);'
    all_test_jobs_num_pattern = r'all_test_jobs_num:\s*(?P<all_test_jobs_num>[-+]?\d+(?:\.\d+)?)'
    all_train_loss_pattern = r'all_train_loss:\s*(?P<all_train_loss>[-+]?\d+(?:\.\d+)?)'
    all_train_accuracy_pattern = r'all_train_accuracy:\s*(?P<all_train_accuracy>[-+]?\d+(?:\.\d+)?)'
    all_test_loss_pattern = r'all_test_loss:\s*(?P<all_test_loss>[-+]?\d+(?:\.\d+)?)'
    all_test_accuracy_pattern = r'all_test_accuracy:\s*(?P<all_test_accuracy>[-+]?\d+(?:\.\d+)?)'
    all_final_significance_pattern = r'all_final_significance:\s*(?P<all_final_significance>[-+]?\d+(?:\.\d+)?)'

    all_target_datablock_num_pattern = r'all_target_datablock_num:\s*(?P<all_target_datablock_num>[-+]?\d+(?:\.\d+)?)'
    all_success_datablock_num_pattern = r'all_success_datablock_num:\s*(?P<all_success_datablock_num>[-+]?\d+(?:\.\d+)?)'
    all_failed_datablock_num_pattern = r'all_failed_datablock_num:\s*(?P<all_failed_datablock_num>[-+]?\d+(?:\.\d+)?)'

    all_need_iter_paths = []
    for file_dir in os.listdir(trace_save_path):
        if "DL_dispatcher" in file_dir:
            sched_file_dir = file_dir.replace("DL_dispatcher", "DL_sched")
            result_dispatcher_file_dir = os.path.join(trace_save_path, file_dir)
            result_sched_file_dir = os.path.join(trace_save_path, sched_file_dir)
            all_need_iter_paths.append([result_dispatcher_file_dir, result_sched_file_dir])

    success_num_arr = []
    failed_num_arr = []

    all_test_jobs_num_arr = []
    all_train_loss_arr = []
    all_train_accuracy_arr = []
    all_test_loss_arr = []
    all_test_accuracy_arr = []
    all_final_significance_arr = []
    all_target_datablock_num_arr = []
    all_success_datablock_num_arr = []
    all_failed_datablock_num_arr = []

    final_used_num = 0
    
    for file_paths in all_need_iter_paths:
        dispatcher_file_path = file_paths[0]
        sched_file_path = file_paths[1]

        match_flags = {
            "current_success_num": False, 
            "all_test_jobs_num": False,
            "all_train_loss": False,
            "all_train_accuracy": False,
            "all_test_loss": False,
            "all_test_accuracy": False,
            "all_final_significance": False,
            "all_target_datablock_num": False,
            "all_success_datablock_num": False,
            "all_failed_datablock_num": False
        }

        with open(dispatcher_file_path, "r+") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if "current_success_num" in line:
                    match = re.search(success_fail_num_pattern, line)
                    if match:
                        success = int(match.group('success'))
                        failed = int(match.group('failed'))
                        success_num_arr.append(success)
                        failed_num_arr.append(failed)
                        match_flags["current_success_num"] = True
                    else:
                        print('current_success_num No match')
                if "all_test_jobs_num" in line:
                    match = re.search(all_test_jobs_num_pattern, line)
                    if match:
                        all_test_jobs_num = int(match.group('all_test_jobs_num'))
                        all_test_jobs_num_arr.append(all_test_jobs_num)
                        match_flags["all_test_jobs_num"] = True
                    else:
                        print('all_test_jobs_num No match')
                if "all_train_loss" in line:
                    match = re.search(all_train_loss_pattern, line)
                    if match:
                        all_train_loss = float(match.group('all_train_loss'))
                        all_train_loss_arr.append(all_train_loss)
                        match_flags["all_train_loss"] = True
                    else:
                        print('all_train_loss No match')
                if "all_train_accuracy" in line:
                    match = re.search(all_train_accuracy_pattern, line)
                    if match:
                        all_train_accuracy = float(match.group('all_train_accuracy'))
                        all_train_accuracy_arr.append(all_train_accuracy)
                        match_flags["all_train_accuracy"] = True
                    else:
                        print('all_train_accuracy No match')
                if "all_test_loss" in line:
                    match = re.search(all_test_loss_pattern, line)
                    if match:
                        all_test_loss = float(match.group('all_test_loss'))
                        all_test_loss_arr.append(all_test_loss)
                        match_flags["all_test_loss"] = True
                    else:
                        print('all_test_loss No match')
                if "all_test_accuracy" in line:
                    match = re.search(all_test_accuracy_pattern, line)
                    if match:
                        all_test_accuracy = float(match.group('all_test_accuracy'))
                        all_test_accuracy_arr.append(all_test_accuracy)
                        match_flags["all_test_accuracy"] = True
                    else:
                        print('all_test_accuracy No match')
                if "all_final_significance" in line:
                    match = re.search(all_final_significance_pattern, line)
                    if match:
                        all_final_significance = float(match.group('all_final_significance'))
                        all_final_significance_arr.append(all_final_significance)
                        match_flags["all_final_significance"] = True
                    else:
                        print('all_final_significance No match')
                if "all_target_datablock_num" in line:
                    match = re.search(all_target_datablock_num_pattern, line)
                    if match:
                        all_target_datablock_num = float(match.group('all_target_datablock_num'))
                        all_target_datablock_num_arr.append(all_target_datablock_num)
                        match_flags["all_target_datablock_num"] = True
                    else:
                        print('all_target_datablock_num No match')
                if "all_success_datablock_num" in line:
                    match = re.search(all_success_datablock_num_pattern, line)
                    if match:
                        all_success_datablock_num = float(match.group('all_success_datablock_num'))
                        all_success_datablock_num_arr.append(all_success_datablock_num)
                        match_flags["all_success_datablock_num"] = True
                    else:
                        print('all_success_datablock_num No match')
                if "all_failed_datablock_num" in line:
                    match = re.search(all_failed_datablock_num_pattern, line)
                    if match:
                        all_failed_datablock_num = float(match.group('all_failed_datablock_num'))
                        all_failed_datablock_num_arr.append(all_failed_datablock_num)
                        match_flags["all_failed_datablock_num"] = True
                    else:
                        print('all_failed_datablock_num No match')
    
        if all(list(match_flags.values())):
            final_used_num += 1
    return final_used_num, success_num_arr, failed_num_arr, all_test_jobs_num_arr, \
            all_train_loss_arr, all_train_accuracy_arr, \
            all_test_loss_arr, all_test_accuracy_arr, all_final_significance_arr, \
            all_target_datablock_num_arr, all_success_datablock_num_arr, all_failed_datablock_num_arr

def read_DL_dispatcher_result_func_simulation_old_version(trace_save_path):
    
    success_fail_num_pattern = r'current_success_num:\s*(?P<success>[-+]?\d+(?:\.\d+)?);\s+current_failed_num:\s*(?P<failed>[-+]?\d+(?:\.\d+)?);\s+current_no_submit_num:\s*(?P<no_submit>[-+]?\d+(?:\.\d+)?);\s+current_no_sche_num:\s*(?P<no_sche>[-+]?\d+(?:\.\d+)?);'
    all_final_significance_pattern = r'all_final_significance:\s*(?P<all_final_significance>[-+]?\d+(?:\.\d+)?)'
    success_final_significance_pattern = r'success_final_significance:\s*(?P<success_final_significance>[-+]?\d+(?:\.\d+)?)'
    old_final_selected_pattern = r'from policy (\[(?P<policy_name>(.*?))\]) selected_datablock_identifiers: (?P<selected_list>\[.*?\])'
    final_selected_pattern = r'from policy (\[(?P<policy_name>(.*?))\]) selected_datablock_identifiers: (?P<selected_map>\{.*?\})'
    failed_selected_pattern = r'failed job scheduling \[(?P<job_id>.+)\]'
    add_new_job_pattern = r'success add new jobs: (?P<job_detail>.+)'
    
    all_test_job_num_pattern = r'dispatcher init job_all_seq_num: (?P<test_job_num>[-+]?\d+(?:\.\d+)?)'

    all_need_iter_paths = []
    for file_dir in os.listdir(trace_save_path):
        if "DL_dispatcher" in file_dir:
            sched_file_dir = file_dir.replace("DL_dispatcher", "DL_sched")
            result_dispatcher_file_dir = os.path.join(trace_save_path, file_dir)
            result_sched_file_dir = os.path.join(trace_save_path, sched_file_dir)
            all_need_iter_paths.append([result_dispatcher_file_dir, result_sched_file_dir])

    success_num_arr = []
    failed_num_arr = []
    all_final_significance_arr = []
    success_final_significance_arr = []
    success_datablock_num_arr = []
    failed_datablock_num_arr = []
    target_datablock_num_arr = []

    final_used_num = 0
    for file_paths in all_need_iter_paths:
        dispatcher_file_path = file_paths[0]
        sched_file_path = file_paths[1]

        final_job_details_map = {}
        job_result_selected_blocks_map = {}
        policy_name_set = set()
        
        job_target_blocknum_map = {}
        job_result_selected_blocknum_map = {}
        all_test_job_num = 0

        match_flags = {
            "current_success_num": False, 
            "all_final_significance": False,
            "success_final_significance": False,
            "job_detail": False,
        }
        with open(dispatcher_file_path, "r+") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if "current_success_num" in line:
                    match = re.search(success_fail_num_pattern, line)
                    if match:
                        success = int(match.group('success'))
                        failed = int(match.group('failed'))
                        success_num_arr.append(success)
                        failed_num_arr.append(failed)
                        match_flags["current_success_num"] = True
                    else:
                        print('No match')
                if "all_final_significance" in line:
                    match = re.search(all_final_significance_pattern, line)
                    if match:
                        all_final_significance = float(match.group('all_final_significance'))
                        all_final_significance_arr.append(all_final_significance)
                        match_flags["all_final_significance"] = True
                    else:
                        print('No match')
                if "success_final_significance" in line:
                    match = re.search(success_final_significance_pattern, line)
                    if match:
                        success_final_significance = float(match.group('success_final_significance'))
                        success_final_significance_arr.append(success_final_significance)
                        match_flags["success_final_significance"] = True
                    else:
                        print('No match')
                
        with open(sched_file_path, "r+") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if "dispatcher init job_all_seq_num" in line:
                    match = re.search(all_test_job_num_pattern, line)
                    if match:
                        all_test_job_num = int(match.group('test_job_num'))
                    else:
                        print('No match')
                if "selected_datablock_identifiers" in line:
                    match = re.search(final_selected_pattern, line)
                    if match:
                        policy_name = match.group('policy_name')
                        policy_name_set.add(policy_name)
                        datablock_identifiers = match.group('selected_map')
                        datablock_identifiers_map = ast.literal_eval(datablock_identifiers)
                        # print(f"temp_datablock_identifiers_list: {datablock_identifiers_list}")
                        for job_id, selected_list in datablock_identifiers_map.items():
                            job_result_selected_blocks_map[job_id] = selected_list
                    else:
                        print('selected_datablock_identifiers No match')
                    # match = re.search(final_selected_pattern, line)
                    # if match:
                    #     policy_name = match.group('policy_name')
                    #     policy_name_set.add(policy_name)
                    #     datablock_identifiers = match.group('selected_list')
                    #     datablock_identifiers_list = ast.literal_eval(datablock_identifiers)
                    #     # print(f"temp_datablock_identifiers_list: {datablock_identifiers_list}")
                    #     for item in datablock_identifiers_list:
                    #         if item[0] not in job_result_selected_blocks_map:
                    #             job_result_selected_blocks_map[item[0]] = set()
                    #         else:
                    #             job_result_selected_blocks_map[item[0]].add(item[1])
                    # else:
                    #     print('selected_datablock_identifiers No match')
                if "failed job scheduling" in line:
                    match = re.search(failed_selected_pattern, line)
                    if match:
                        failed_job_id = match.group('job_id')
                        job_result_selected_blocks_map[failed_job_id] = []
                    else:
                        print('failed job scheduling No match')
                if "success add new jobs" in line:
                    match = re.search(add_new_job_pattern, line)
                    if match:
                        job_detail_map = match.group('job_detail')
                        job_detail_map = ast.literal_eval(job_detail_map)
                        # print(f"temp_job_detail_map: {job_detail_map}")
                        for key, value in job_detail_map.items():
                            final_job_details_map[key] = value
                            job_target_blocknum_map[key] = value['datablock_select_num']
                    else:
                        print('success add new jobs No match')
        
        # print(f"len job_result_selected_blocks_map: {len(job_result_selected_blocks_map)}")
        # print(f"len job_target_blocknum_map: {len(job_target_blocknum_map)}")
        # print(f"len final_job_details_map: {len(final_job_details_map)}")
        # print(f"all_test_job_num: {all_test_job_num}")

        if len(job_result_selected_blocks_map) == all_test_job_num and \
            len(job_target_blocknum_map) == all_test_job_num and \
            len(final_job_details_map) == all_test_job_num:
            match_flags["job_detail"] = True
        # print(f"match_flags: {match_flags}")
        if all(list(match_flags.values())):
            final_used_num += 1

            for job_id, selected_block_set in job_result_selected_blocks_map.items():
                job_result_selected_blocknum_map[job_id] = len(selected_block_set)
            
            temp_success_num = 0
            temp_target_num = 0
            for job_id in job_target_blocknum_map:
                temp_success_num += job_result_selected_blocknum_map[job_id]
                temp_target_num += job_target_blocknum_map[job_id]
            success_datablock_num_arr.append(temp_success_num)
            target_datablock_num_arr.append(temp_target_num)
            failed_datablock_num_arr.append(temp_target_num - temp_success_num)
    
    # print(f"final_used_num: {final_used_num}")
    # print(f"success_num_arr: {success_num_arr}")
    # print(f"failed_num_arr: {failed_num_arr}")
    # print(f"all_final_significance_arr: {all_final_significance_arr}")
    # print(f"success_final_significance_arr: {success_final_significance_arr}")
    # print(f"success_datablock_num_arr: {success_datablock_num_arr}")
    # print(f"failed_datablock_num_arr: {failed_datablock_num_arr}")

    return final_used_num, success_num_arr, failed_num_arr, \
            all_final_significance_arr, success_final_significance_arr, \
            success_datablock_num_arr, failed_datablock_num_arr, target_datablock_num_arr
            