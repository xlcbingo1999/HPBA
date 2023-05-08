import os
import re
import numpy as np
from utils.global_variable import RESULT_PATH

def read_DL_dispatcher_result_func(trace_save_path):
    all_need_iter_paths = []
    for file_dir in os.listdir(trace_save_path):
        if "DL_dispatcher" in file_dir:
            result_read_file_dir = os.path.join(trace_save_path, file_dir)
            all_need_iter_paths.append(result_read_file_dir)
    success_fail_num_pattern = r'current_success_num:\s*(?P<success>\d+);\s+current_failed_num:\s*(?P<failed>\d+);\s+current_no_submit_num:\s*(?P<no_submit>\d+);\s+current_no_sche_num:\s*(?P<no_sche>\d+);'
    all_final_significance_pattern = r'all_final_significance:\s*(?P<all_final_significance>\d+\.\d+)'
    success_final_significance_pattern = r'success_final_significance:\s*(?P<success_final_significance>\d+\.\d+)'

    success_num = []
    failed_num = []
    all_final_significance_arr = []
    success_final_significance_arr = []
    final_used_num = 0
    for file_path in all_need_iter_paths:
        match_flags = {
            "current_success_num": False, 
            "all_final_significance": False,
            "success_final_significance": False
        }
        with open(file_path, "r+") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if "current_success_num" in line:
                    match = re.search(success_fail_num_pattern, line)
                    if match:
                        success = int(match.group('success'))
                        failed = int(match.group('failed'))
                        success_num.append(success)
                        failed_num.append(failed)
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
        
        if all(list(match_flags.values())):
            final_used_num += 1
    return final_used_num, success_num, failed_num, all_final_significance_arr, success_final_significance_arr