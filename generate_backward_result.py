import os
import re
import numpy as np
from utils.global_variable import RESULT_PATH
from utils.data_operator import read_DL_dispatcher_result_func

def final_operate_data(current_test_all_dir):
    print("current_test_all_dir")
    trace_save_path = "{}/{}".format(RESULT_PATH, current_test_all_dir)
    final_used_num, success_num_arr, failed_num_arr, all_final_significance_arr, success_final_significance_arr = read_DL_dispatcher_result_func(trace_save_path)
    
    # 新建一个全新的log进行保存
    all_result_path = "{}/all_result.log".format(trace_save_path)
    print("all_result_path: ", all_result_path)
    with open(all_result_path, "w+") as f:
        print("final_used_num: {}".format(final_used_num))
        print("final_used_num: {}".format(final_used_num), file=f)
        print("[success_info] {}({}-{}) === success_num_mean: {} ; success_num_min: {} ; success_num_max: {}".format(
            np.mean(success_num_arr), min(success_num_arr), max(success_num_arr), np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
        ))
        print("[success_info] {}({}-{}) === success_num_mean: {} ; success_num_min: {} ; success_num_max: {}".format(
            np.mean(success_num_arr), min(success_num_arr), max(success_num_arr), np.mean(success_num_arr), min(success_num_arr), max(success_num_arr)
        ), file=f)
        print("[failed_info] {}({}-{}) === failed_num_mean: {} ; failed_num_min: {} ; failed_num_max: {}".format(
            np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr), np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
        ))
        print("[failed_info] {}({}-{}) === failed_num_mean: {} ; failed_num_min: {} ; failed_num_max: {}".format(
            np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr), np.mean(failed_num_arr), min(failed_num_arr), max(failed_num_arr)
        ), file=f)
        print("[allsig_info] {}({}-{}) === all_final_significance_mean: {} ; all_final_significance_min: {} ; all_final_significance_max: {}".format(
            np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr), np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
        ))
        print("[allsig_info] {}({}-{}) === all_final_significance_mean: {} ; all_final_significance_min: {} ; all_final_significance_max: {}".format(
            np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr), np.mean(all_final_significance_arr), min(all_final_significance_arr), max(all_final_significance_arr)
        ), file=f)
        print("[successsig_info] {}({}-{}) === success_final_significance_mean: {} ; success_final_significance_min: {} ; success_final_significance_max: {}".format(
            np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr), np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr)
        ))
        print("[successsig_info] {}({}-{}) === success_final_significance_mean: {} ; success_final_significance_min: {} ; success_final_significance_max: {}".format(
            np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr), np.mean(success_final_significance_arr), min(success_final_significance_arr), max(success_final_significance_arr)
        ), file=f)


all_current_test_all_dirs = [
    "schedule-review-simulation-05-09-22-21-01",
    "schedule-review-simulation-05-09-22-28-37",
    "schedule-review-simulation-05-09-22-29-52",
    "schedule-review-simulation-05-09-22-30-36",
    "schedule-review-simulation-05-09-23-11-03",
    "schedule-review-simulation-05-09-23-12-38",
    "schedule-review-simulation-05-09-23-15-37",
    "schedule-review-simulation-05-09-23-16-51",
    "schedule-review-simulation-05-09-23-19-46",
    "schedule-review-simulation-05-09-23-20-22",
    "schedule-review-simulation-05-09-23-11-03",
    "schedule-review-simulation-05-09-23-25-58",
    "schedule-review-simulation-05-09-23-26-45",
    "schedule-review-simulation-05-09-23-27-35"
]
for dir in all_current_test_all_dirs:
    final_operate_data(dir)