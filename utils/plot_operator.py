import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.data_operator import is_number

def get_mark_color_hatch_marker():
    colors = ["#3b6291", "#943c39", "#779043", "#624c7c", "#388498", "#bf7334", "#3f6899", "#9c403d",
            "#7d9847", "#675083", "#3b8ba1", "#c97937"]
    hatchs = ['-', 'x', '/', '*', '\\\\', '+', 'o', '.']
    markers = ['x', 'o', 'v', '^', '<', '>', 's', 'P', 'X', 'D']
    return colors, hatchs, markers

def add_df_with_min_max(df):
    add_columns_keys = [
        "Success num", 
        "Failed num", 
        "Mean Significance (ALL)", 
        "Mean Significance (Success)",
        "Success Datablock Num",
        "Failed Datablock Num",
        "Target Datablock Num",
        "Train Loss",	
        "Train Accuracy",	
        "Test Loss",	
        "Test Accuracy"
    ]
    key_need_max = [True, False, True, True, True, False, True]
    # 遍历每一行并进行正则表达式匹配和提取
    for index, row in df.iterrows():
        for key_index, key in enumerate(add_columns_keys):
            success_num = row[key]
            success_num_str = str(success_num)
            # print(f"success_num_str: {success_num}")
            if pd.isnull(success_num) or success_num_str.isspace():
                print(f"success_num_str nan!")
                df.loc[index, f'{key} avg'] = 0.0
                df.loc[index, f'{key} min'] = 0.0
                df.loc[index, f'{key} max'] = 0.0
            elif '[' in success_num_str and ']' in success_num_str:
                all_success_num_strs = success_num_str.split(";")
                print(f"all_success_num_strs: {all_success_num_strs}")
                need_max = key_need_max[key_index]
                nedd_avg = -float("inf") if need_max else float("inf")
                for temp_success_num_str in all_success_num_strs:
                    split_name_kuohao = temp_success_num_str.split("]")
                    split_left_kuohao = split_name_kuohao[1].split("(")
                    avg_value = float(split_left_kuohao[0])
                    split_lianzifu = split_left_kuohao[1].split("~")
                    min_value = float(split_lianzifu[0])
                    max_value = float(split_lianzifu[1].split(")")[0])
                    print(f"temp_success_num_str: {temp_success_num_str} => avg_value: {avg_value}; min_value: {min_value}; max_value: {max_value}")
                    if (need_max and avg_value > nedd_avg) or (not need_max and avg_value < nedd_avg):
                        nedd_avg = avg_value
                        df.loc[index, f'{key} avg'] = avg_value
                        df.loc[index, f'{key} min'] = min_value
                        df.loc[index, f'{key} max'] = max_value
            elif ('(' in success_num_str) and ('~' in success_num_str) and (')' in success_num_str):
                split_left_kuohao = success_num_str.split("(")
                avg_value = float(split_left_kuohao[0])
                split_lianzifu = split_left_kuohao[1].split("~")
                min_value = float(split_lianzifu[0])
                max_value = float(split_lianzifu[1].split(")")[0])
                

                df.loc[index, f'{key} avg'] = avg_value
                df.loc[index, f'{key} min'] = min_value
                df.loc[index, f'{key} max'] = max_value
            elif is_number(success_num_str):
                avg_value = float(success_num_str)
                df.loc[index, f'{key} avg'] = avg_value
                df.loc[index, f'{key} min'] = avg_value
                df.loc[index, f'{key} max'] = avg_value
            else:
                df.loc[index, f'{key} avg'] = 0.0
                df.loc[index, f'{key} min'] = 0.0
                df.loc[index, f'{key} max'] = 0.0
    return df

def get_result_avg_min_max_for_y_label_name(df_with_key, out_loop_groups, in_loop_groups, y_label_name):
    results = [[0.0 for _ in range(len(in_loop_groups))] for _ in range(len(out_loop_groups))] 
    results_min = [[0.0 for _ in range(len(in_loop_groups))] for _ in range(len(out_loop_groups))] 
    results_max = [[0.0 for _ in range(len(in_loop_groups))] for _ in range(len(out_loop_groups))] 

    for out_index, out_key in enumerate(out_loop_groups):
        for in_index, in_key in enumerate(in_loop_groups):
            test_job_num = int(df_with_key.loc[(out_key, in_key), "Online job num"])
            if y_label_name == "Number of Allocated Jobs":
                success_key_prefix = "Success num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} max"]
            elif y_label_name == "Number of Failed Jobs":
                failed_key_prefix = "Failed num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{failed_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{failed_key_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{failed_key_prefix} max"]
            elif y_label_name == "Ratio of Allocated Jobs":
                success_key_prefix = "Success num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"] / test_job_num
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} min"] / test_job_num
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} max"] / test_job_num
            elif y_label_name == "Ratio of Allocated Datablocks":
                success_key_prefix = "Success Datablock Num"
                target_key_prefix = "Target Datablock Num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"] / df_with_key.loc[(out_key, in_key), f"{target_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} min"] / df_with_key.loc[(out_key, in_key), f"{target_key_prefix} avg"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} max"] / df_with_key.loc[(out_key, in_key), f"{target_key_prefix} avg"]
            elif y_label_name == "Average Significance of all jobs":
                avg_sig_all_job_key_prefix = "Mean Significance (ALL)"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} max"]
            elif y_label_name == "Average Significance of allocated jobs":
                avg_sig_success_job_key_prefix = "Mean Significance (Success)"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} max"]
            elif y_label_name == "Significance of all jobs":
                avg_sig_all_job_key_prefix = "Mean Significance (ALL)"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} avg"] * test_job_num
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} min"] * test_job_num
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} max"] * test_job_num
            elif y_label_name == "Significance of allocated jobs":
                avg_sig_success_job_key_prefix = "Mean Significance (Success)"
                success_key_prefix = "Success num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} avg"] * df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} min"] * df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} max"] * df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
            elif (y_label_name == "Train Loss" or y_label_name == "Train Accuracy" or 
                y_label_name == "Test Loss" or y_label_name == "Test Accuracy") :
                train_accuracy_prefix = y_label_name
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_accuracy_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_accuracy_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_accuracy_prefix} max"]

    print("results: {}".format(results))
    print("results_min: {}".format(results_min))
    print("results_max: {}".format(results_max))
    return results, results_min, results_max