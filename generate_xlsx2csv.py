import pandas as pd
import numpy as np
import csv
import os

file_name = 'testbed_fig_3' # temp_get_result, 20230510_privacy_new, testbed_fig_1

sheet_name_2_result_file_name = {
    # "Sheet1": "testbed_Q1"
    "Sheet1": "testbed_fig_3",
    # "Sheet1": "testbed_fig_2",
    # "fig2_max_min_ratio": "fig_2",
    # "fig1_online_job_num": "fig_1",
    # "fig6_HIS_history_change": "fig_6",
    # "fig5_batch_size": "fig_5",
    # "Sheet1": "temp_get_result",
    # "Sheet1": "testbed_fig_1",
}
# 定义输入的 Excel 文件路径
plots_path = "/home/netlab/DL_lab/opacus_testbed/plots"
input_excel_file = os.path.join(plots_path, '{}.xlsx'.format(file_name)) 

# 定义输出的 CSV 文件路径
for sheet_name, result_file_name in sheet_name_2_result_file_name.items():
    output_csv_file = os.path.join(plots_path, '{}.csv'.format(result_file_name))

    # 读取 Excel 文件中的数据
    data_frame = pd.read_excel(input_excel_file, sheet_name=sheet_name)

    # 去除所有格式
    data_frame = data_frame.applymap(str)
    data_frame = data_frame.replace({
        r'^"|"$': '', 
        r"^'|'$": '',
        "\n": ' ', 
        pd.notna: ''
    }, regex=True)
    data_frame = data_frame.applymap(lambda x: x.strip() if isinstance(x, str) else x)


    # 将数据保存为 CSV 文件
    data_frame.to_csv(output_csv_file, index=False)
