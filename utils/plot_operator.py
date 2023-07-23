import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.data_operator import is_number

# (浅红)ffadad-(浅橙)ffd6a5-(浅黄)fdffb6-(浅绿)caffbf-(浅天蓝)9bf6ff-(浅海蓝)a0c4ff-(浅紫)bdb2ff-(浅粉)ffc6ff-(浅白)fffffc
# (深黑)001219-(深海蓝)005f73-(深墨绿)0a9396-(深天蓝)94d2bd-(深白)e9d8a6-(深橙)ee9b00-(深红)ca6702-(非常深红)9b2226

'''
# (浅红)ffadad-(浅橙)ffd6a5-(浅黄)fdffb6-(浅绿)caffbf-(浅天蓝)9bf6ff-(浅海蓝)a0c4ff-(浅紫)bdb2ff-(浅粉)ffc6ff-(浅白)fffffc
# (深黑)001219-(深海蓝)005f73-(深墨绿)0a9396-(深天蓝)94d2bd-(深白)e9d8a6-(深橙)ee9b00-(深红)ca6702-(非常深红)9b2226
def get_mark_color_hatch_marker():
        colors =["#ffadad", "#ffd6a5", 
                "#fdffb6",  "#caffbf", 
                "#9bf6ff",  "#bdb2ff", 
                "#ffc6ff", "#a0c4ff"]
        hatchs = ['-', '*', 
                '/', 'o', 
                '\\\\', '.',
                '////', '+*', ]
        markers = ['x', 'o', 
                    'v', '^', 
                    '<', '>', 
                    'P', 's']
        return colors, hatchs, markers
'''

def add_df_with_min_max(df, add_columns_keys_2_need_max_map):
    # 遍历每一行并进行正则表达式匹配和提取
    for index, row in df.iterrows():
        for key, key_need_max in add_columns_keys_2_need_max_map.items():
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
                need_max = key_need_max
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
                test_job_num = int(df_with_key.loc[(out_key, in_key), "Test Jobs Num"])
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"] / test_job_num
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} min"] / test_job_num
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} max"] / test_job_num
            elif y_label_name == "Ratio of Allocated Datablocks":
                success_key_prefix = "Success Datablock Num"
                target_key_prefix = "Target Datablock Num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"] / df_with_key.loc[(out_key, in_key), f"{target_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} min"] / df_with_key.loc[(out_key, in_key), f"{target_key_prefix} avg"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{success_key_prefix} max"] / df_with_key.loc[(out_key, in_key), f"{target_key_prefix} avg"]
            elif y_label_name == "Significance of all queries" or y_label_name == "Total Significances":
                avg_sig_all_job_key_prefix = "Mean Significance All"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_all_job_key_prefix} max"]
            elif y_label_name == "Significance of allocated queries":
                avg_sig_success_job_key_prefix = "Mean Significance Success"
                success_key_prefix = "Success num"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} avg"] * df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} min"] * df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{avg_sig_success_job_key_prefix} max"] * df_with_key.loc[(out_key, in_key), f"{success_key_prefix} avg"]
            elif y_label_name == "Sum of Delta Train Accuracy":
                train_acc_prefix = "Train Accuracy All"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_acc_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_acc_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_acc_prefix} max"]
            elif y_label_name == "Sum of Delta Train Loss":
                train_loss_prefix = "Train Loss All"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_loss_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_loss_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_loss_prefix} max"]
            elif y_label_name == "Sum of Delta Test Accuracy" or y_label_name == "Total Test Accuracy Improvement (%)":
                test_acc_prefix = "Test Accuracy All"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{test_acc_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{test_acc_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{test_acc_prefix} max"]
            elif y_label_name == "Sum of Delta Test Loss":
                test_loss_prefix = "Test Loss All"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{test_loss_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{test_loss_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{test_loss_prefix} max"]
            elif (y_label_name == "Epsilon_Real_All_Block" or y_label_name == "Significance_Epsilon_Ratio" or 
                y_label_name == "Test_Loss_Epsilon_Ratio" or y_label_name == "Test_Accuracy_Epsilon_Ratio") :
                train_accuracy_prefix = y_label_name
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_accuracy_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_accuracy_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{train_accuracy_prefix} max"]
            elif (y_label_name == "Average Decision Time Consumption (s)"):
                time_prefix = "Decision_Duration"
                results[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{time_prefix} avg"]
                results_min[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{time_prefix} min"]
                results_max[out_index][in_index] = df_with_key.loc[(out_key, in_key), f"{time_prefix} max"]

    print("results: {}".format(results))
    print("results_min: {}".format(results_min))
    print("results_max: {}".format(results_max))
    return results, results_min, results_max