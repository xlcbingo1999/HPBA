import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def add_df_with_min_max(df):
    add_columns_keys = ["Success num", "Failed num", "Mean Significance (ALL)", "Mean Significance (Success)"]
    # 遍历每一行并进行正则表达式匹配和提取
    for index, row in df.iterrows():
        for key in add_columns_keys:
            success_num_str = row[key]
            print(f"success_num_str: {success_num_str} => len(success_num_str): {len(success_num_str)}")
            if ('(' in success_num_str) and ('-' in success_num_str) and (')' in success_num_str):
                split_left_kuohao = success_num_str.split("(")
                avg_value = float(split_left_kuohao[0])
                split_lianzifu = split_left_kuohao[1].split("-")
                min_value = float(split_lianzifu[0])
                max_value = float(split_lianzifu[1].split(")")[0])
                

                df.loc[index, f'{key} avg'] = avg_value
                df.loc[index, f'{key} min'] = min_value
                df.loc[index, f'{key} max'] = max_value
            elif success_num_str.isdigit():
                avg_value = float(success_num_str)
                df.loc[index, f'{key} avg'] = avg_value
                df.loc[index, f'{key} min'] = avg_value
                df.loc[index, f'{key} max'] = avg_value
            else:
                df.loc[index, f'{key} avg'] = 0.0
                df.loc[index, f'{key} min'] = 0.0
                df.loc[index, f'{key} max'] = 0.0
    return df

current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
path = current_dir + "/fig_1.csv"
df = pd.read_csv(path)

keys_str = ["policy", "Online job num"]
df_with_key = df.set_index(keys_str)
unique_values = df_with_key.index.unique()
print(unique_values)

df_with_key = add_df_with_min_max(df_with_key)
print("---- df.info ----")
df_with_key.info()
test_jobs_num_groups = [1000, 2000, 3000, 4000]
policy_groups = {
    "IterativeHISwithOrderPolicy(100)": "IterativeHIS", 
    # "HISPolicy": "HIS", 
    "PBGPolicy": "PBG",
    "PBGMixPolicy": "PBGMix", 
    "SagewithRemainPolicy": "Sage",
    "BestFitwithRemainPolicy": "BestFit",
    "OfflinePolicy": "Ground Truth"
}
marks = ['o'] * len(policy_groups)
colors = ["#3b6291", "#943c39", "#779043", "#624c7c", "#388498", "#bf7334", "#3f6899", "#9c403d",
        "#7d9847", "#675083", "#3b8ba1", "#c97937"]
hatchs = ['-', '/', '\\\\', 'x', '*', '+']

y_label_name = "Significance of allocated jobs" # "Number of Allocated Jobs", "Ratio of Allocated Jobs", "Significance of all jobs", "Significance of allocated jobs"
results = [[0.0 for _ in range(len(test_jobs_num_groups))] for _ in range(len(policy_groups))] 
results_min = [[0.0 for _ in range(len(test_jobs_num_groups))] for _ in range(len(policy_groups))] 
results_max = [[0.0 for _ in range(len(test_jobs_num_groups))] for _ in range(len(policy_groups))] 

for policy_index, policy in enumerate(policy_groups):
    for group_index, test_job_num in enumerate(test_jobs_num_groups):
        if y_label_name == "Number of Allocated Jobs":
            success_key_prefix = "Success num"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} avg"]
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} min"]
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} max"]
        elif y_label_name == "Number of Failed Jobs":
            failed_key_prefix = "Failed num"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{failed_key_prefix} avg"]
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{failed_key_prefix} min"]
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{failed_key_prefix} max"]
        elif y_label_name == "Ratio of Allocated Jobs":
            success_key_prefix = "Success num"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} avg"] / test_job_num
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} min"] / test_job_num
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} max"] / test_job_num
        elif y_label_name == "Average Significance of all jobs":
            avg_sig_all_job_key_prefix = "Mean Significance (ALL)"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_all_job_key_prefix} avg"]
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_all_job_key_prefix} min"]
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_all_job_key_prefix} max"]
        elif y_label_name == "Average Significance of allocated jobs":
            avg_sig_success_job_key_prefix = "Mean Significance (Success)"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_success_job_key_prefix} avg"]
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_success_job_key_prefix} min"]
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_success_job_key_prefix} max"]
        elif y_label_name == "Significance of all jobs":
            avg_sig_all_job_key_prefix = "Mean Significance (ALL)"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_all_job_key_prefix} avg"] * test_job_num
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_all_job_key_prefix} min"] * test_job_num
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_all_job_key_prefix} max"] * test_job_num
        elif y_label_name == "Significance of allocated jobs":
            avg_sig_success_job_key_prefix = "Mean Significance (Success)"
            success_key_prefix = "Success num"
            results[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_success_job_key_prefix} avg"] * df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} avg"]
            results_min[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_success_job_key_prefix} min"] * df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} avg"]
            results_max[policy_index][group_index] = df_with_key.loc[(policy, test_job_num), f"{avg_sig_success_job_key_prefix} max"] * df_with_key.loc[(policy, test_job_num), f"{success_key_prefix} avg"]


print("results: {}".format(results))
print("results_min: {}".format(results_min))
print("results_max: {}".format(results_max))

fig = plt.figure(figsize=(10, 5))
plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

bar_width = 0.2
ratio = 2
font_size = 20
henzuobiao_indexes = np.arange(len(test_jobs_num_groups)) * ratio

# for policy_index, policy in enumerate(policy_groups):
#     plt.plot(test_jobs_num_groups, results[policy_index], marker=marks[policy_index], color=colors[policy_index], label=policy_groups[policy], linewidth=1.5)
#     plt.fill_between(test_jobs_num_groups, results_min[policy_index], results_max[policy_index], color=colors[policy_index], alpha=0.5)

for policy_index, policy in enumerate(policy_groups):
    print(f"policy_index: {policy_index}: policy: {policy}")
    print(f"results[group_index]: {results[group_index]}")
    plt.bar(
        henzuobiao_indexes + policy_index * bar_width, 
        results[policy_index], 
        bar_width, 
        color=colors[policy_index], 
        label=policy_groups[policy],
        hatch=hatchs[policy_index],
        edgecolor="black", 
    )

group_labels = list(str(hen) for hen in test_jobs_num_groups)  # x轴刻度的标识
center_ratio = 2.5
plt.xticks(
    henzuobiao_indexes + center_ratio * bar_width, 
    group_labels, 
    fontsize=font_size, 
    fontweight='bold'
)  # 默认字体大小为10
plt.yticks(fontsize=font_size, fontweight='bold')
plt.xlabel(r"Number of test jobs $n$", fontsize=font_size, fontweight='bold')
plt.ylabel(y_label_name, fontsize=font_size, fontweight='bold')
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

# plt.xlim(0.9, 6.1)  # 设置x轴的范围
# plt.ylim(1.5, 16)

plt.legend()          #显示各曲线的图例
# plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细
# plt.legend(loc=4, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
legend_properties = {
    'weight':'bold',
    'size': font_size
}
plt.legend(bbox_to_anchor=(0.5,1.22), columnspacing=0.4, loc='upper center', ncol=3, prop=legend_properties, frameon=False)

plt.subplots_adjust(left=0.1, right=0.88)

plt.tight_layout()
result_path_prefix = current_dir + "/fig_1_{}".format(y_label_name)
plt.savefig(result_path_prefix + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(result_path_prefix + '.pdf')
pp.savefig(fig)
pp.close()