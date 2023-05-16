import os
import re
import ast
import numpy as np
from utils.global_variable import RESULT_PATH

def read_DL_dispatcher_result_func(trace_save_path):
    
    success_fail_num_pattern = r'current_success_num:\s*(?P<success>\d+);\s+current_failed_num:\s*(?P<failed>\d+);\s+current_no_submit_num:\s*(?P<no_submit>\d+);\s+current_no_sche_num:\s*(?P<no_sche>\d+);'
    all_final_significance_pattern = r'all_final_significance:\s*(?P<all_final_significance>\d+\.\d+)'
    success_final_significance_pattern = r'success_final_significance:\s*(?P<success_final_significance>\d+\.\d+)'
    final_selected_pattern = r'from policy (\[(?P<policy_name>(.*?))\]) selected_datablock_identifiers: (?P<selected_list>\[.*?\])'
    failed_selected_pattern = r'failed job scheduling \[(?P<job_id>.+)\]'
    add_new_job_pattern = r'success add new jobs: (?P<job_detail>.+)'
    
    all_test_job_num_pattern = r'dispatcher init job_all_seq_num: (?P<test_job_num>\d+)'

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
                        datablock_identifiers = match.group('selected_list')
                        datablock_identifiers_list = ast.literal_eval(datablock_identifiers)
                        # print(f"temp_datablock_identifiers_list: {datablock_identifiers_list}")
                        for item in datablock_identifiers_list:
                            if item[0] not in job_result_selected_blocks_map:
                                job_result_selected_blocks_map[item[0]] = set()
                            else:
                                job_result_selected_blocks_map[item[0]].add(item[1])
                    else:
                        print('selected_datablock_identifiers No match')
                if "failed job scheduling" in line:
                    match = re.search(failed_selected_pattern, line)
                    if match:
                        failed_job_id = match.group('job_id')
                        job_result_selected_blocks_map[failed_job_id] = set()
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
            