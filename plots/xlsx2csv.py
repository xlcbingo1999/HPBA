import pandas as pd

file_name = '20230510_privacy_new'
sheet_name = 'fig1_online_job_num'
# 定义输入的 Excel 文件路径
input_excel_file = '{}.xlsx'.format(file_name)

# 定义输出的 CSV 文件路径
output_csv_file = '{}.csv'.format(file_name)

# 读取 Excel 文件中的数据
data_frame = pd.read_excel(input_excel_file, sheet_name=sheet_name)

# 去除所有格式
data_frame = data_frame.applymap(str)
data_frame = data_frame.replace({'\n': ' ', '"': ''}, regex=True)

# 将数据保存为 CSV 文件
data_frame.to_csv(output_csv_file, index=False)
