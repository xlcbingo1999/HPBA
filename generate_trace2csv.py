import pandas as pd
import numpy as np
import os
import sys
import re
from utils.data_operator import final_operate_data, is_number


def write_to_df(df, index, success_num_arr, failed_num_arr,
                all_final_significance_arr, 
                success_final_significance_arr,
                success_datablock_num_arr, 
                failed_datablock_num_arr, 
                target_datablock_num_arr,
                run_time):
    df.loc[index, "Success num"] = f"{np.mean(success_num_arr)}({min(success_num_arr)}-{max(success_num_arr)})"
    df.loc[index, "Failed num"] = f"{np.mean(failed_num_arr)}({min(failed_num_arr)}-{max(failed_num_arr)})"
    df.loc[index, "Mean Significance (ALL)"] = f"{np.mean(all_final_significance_arr)}({min(all_final_significance_arr)}-{max(all_final_significance_arr)})"
    df.loc[index, "Mean Significance (Success)"] = f"{np.mean(success_final_significance_arr)}({min(success_final_significance_arr)}-{max(success_final_significance_arr)})"
    df.loc[index, "Success Datablock Num"] = f"{np.mean(success_datablock_num_arr)}({min(success_datablock_num_arr)}-{max(success_datablock_num_arr)})"
    df.loc[index, "Failed Datablock Num"] = f"{np.mean(failed_datablock_num_arr)}({min(failed_datablock_num_arr)}-{max(failed_datablock_num_arr)})"
    df.loc[index, "Target Datablock Num"] = f"{np.mean(target_datablock_num_arr)}({min(target_datablock_num_arr)}-{max(target_datablock_num_arr)})"
    df.loc[index, "Run Time"] = int(run_time)
    return df

def update_df_real(df):
    need_search_columns_keys = [
        "Success num", 
        "Failed num", 
        "Mean Significance (ALL)", 
        "Mean Significance (Success)",
        "Success Datablock Num",
        "Failed Datablock Num",
        "Target Datablock Num",
    ]
    mulu_column_key = "log目录"
    mulu_column_key_format_len = 41
    for index, row in df.iterrows():
        row_update_flag = False
        log_trace_paths = str(row[mulu_column_key]).split("; ")
        # print(f"log_trace_paths: {log_trace_paths}")
        keystr2logmulu = {}
        for path in log_trace_paths:
            match = re.match(r"\[(?P<keystr>.*)\]\s*(?P<logmulu>.*)", path)
            if match:
                keystr = str(match.group("keystr"))
                logmulu = str(match.group("logmulu")).strip()
                keystr2logmulu[keystr] = logmulu
            else:
                keystr2logmulu["default"] = path
        
        print(f"keystr2logmulu: {keystr2logmulu}")

        run_time_item = row["Run Time"]
        run_time_item_str = str(run_time_item)
        if pd.isnull(run_time_item) or run_time_item_str.isspace():
            row_update_flag = True
            print(f"row [{index}] need to update!")
        elif is_number(run_time_item_str) and float(run_time_item_str) < float(row["Target Run Time"]):
            row_update_flag = True
            print(f"row [{index}] need to update!")
        for key_index, key in enumerate(need_search_columns_keys):
            wait_judge_item = row[key]
            wait_judge_item_str = str(wait_judge_item)
            # print(f"wait_judge_item: {wait_judge_item}")
            if pd.isnull(wait_judge_item) or wait_judge_item_str.isspace():
                row_update_flag = True
                print(f"row [{index}] need to update!")
                break
            elif is_number(wait_judge_item_str):
                row_update_flag = True
                print(f"row [{index}] need to update!")
                break
        if row_update_flag:
            log_trace_path = row[mulu_column_key][0:mulu_column_key_format_len]
            log_trace_full_path = os.path.join(trace_dir, log_trace_path)
            if os.path.exists(log_trace_full_path):
                final_used_num, success_num_arr, failed_num_arr, \
                    all_final_significance_arr, success_final_significance_arr, \
                    success_datablock_num_arr, failed_datablock_num_arr, target_datablock_num_arr = final_operate_data(log_trace_path)
                if final_used_num > 0:
                    df = write_to_df(df, index, 
                                    success_num_arr, 
                                    failed_num_arr,
                                    all_final_significance_arr, 
                                    success_final_significance_arr,
                                    success_datablock_num_arr, 
                                    failed_datablock_num_arr, 
                                    target_datablock_num_arr, 
                                    final_used_num)
                    print(f"write success!")
            else:
                print(f"log_trace_full_path [{log_trace_full_path}] no exist")
    return df

if __name__ == "__main__":
    root_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    file_names = ["fig_5"] # temp_get_result, fig_5, fig_1, fig_2, fig_6
    for file_name in file_names:
        target_path = os.path.join(root_dir, f"{file_name}.csv")
        result_path = os.path.join(root_dir, f"{file_name}_right.csv")
        trace_dir = "/mnt/linuxidc_client/opacus_testbed_result"

        df = pd.read_csv(target_path)

        df = update_df_real(df)
        df.to_csv(result_path, index=False)

    
