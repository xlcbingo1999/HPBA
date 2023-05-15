import pandas as pd
import numpy as np

file_name = '20230510_privacy_new'
result_file_name = 'fig_2'
sheet_name = 'fig3_max_min_ratio'
# 定义输入的 Excel 文件路径
input_excel_file = '{}.xlsx'.format(file_name)

# 定义输出的 CSV 文件路径
output_csv_file = '{}.csv'.format(result_file_name)

# 读取 Excel 文件中的数据
data_frame = pd.read_excel(input_excel_file, sheet_name=sheet_name)

# 去除所有格式
data_frame = data_frame.applymap(str)
data_frame = data_frame.replace({'\n': ' ', '"': '', pd.notna: ''}, regex=True)

# 将数据保存为 CSV 文件
data_frame.to_csv(output_csv_file, index=False)
