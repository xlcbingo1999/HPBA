import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_utils import add_df_with_min_max, get_mark_color_hatch, get_result_avg_min_max_for_y_label_name

# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
path = current_dir + "/fig_2.csv"
df = pd.read_csv(path)

keys_str = ["policy", "Max-min Ratio"]
df_with_key = df.set_index(keys_str)
unique_values = df_with_key.index.unique()
print(unique_values)
df_with_key = add_df_with_min_max(df_with_key)
print("---- df.info ----")
df_with_key.info()

max_min_ratio_groups = [0.025, 0.033, 0.05, 0.1, 0.4, 0.7, 1.0]
policy_groups = {
    "OfflinePolicy": "Ground Truth",
    "HISwithOrderProVersionPolicy": "HIS", 
    "IterativeHISwithOrderPolicy(100)": "IterativeHIS", 
    "PBGPolicy": "PBG",
    "PBGMixPolicy": "PBGMix", 
    "SagewithRemainPolicy": "Sage",
    "BestFitwithRemainPolicy": "BestFit"
}
y_label_name_arr = [
    "Ratio of Allocated Jobs", 
    "Significance of all jobs", 
    "Significance of allocated jobs",
    "Ratio of Allocated Datablocks"
] 
# "Number of Allocated Jobs", 
# "Ratio of Allocated Jobs", 
# "Significance of all jobs", 
# "Significance of allocated jobs",
# "Success Datablock Num",
# "Failed Datablock Num",
# "Target Datablock Num"

colors, hatchs = get_mark_color_hatch()
for y_label_name in y_label_name_arr:
    results, results_min, results_max = get_result_avg_min_max_for_y_label_name(df_with_key, policy_groups, max_min_ratio_groups, y_label_name)

    fig = plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    bar_width = 0.23
    ratio = 2
    font_size = 20
    henzuobiao_indexes = np.arange(len(max_min_ratio_groups)) * ratio

    for policy_index, policy in enumerate(policy_groups):
        print(f"policy_index: {policy_index}: policy: {policy}")
        print(f"results[group_index]: {results[policy_index]}")
        plt.bar(
            henzuobiao_indexes + policy_index * bar_width, 
            results[policy_index], 
            bar_width, 
            color=colors[policy_index], 
            label=policy_groups[policy],
            hatch=hatchs[policy_index],
            edgecolor="black", 
        )

    group_labels = list(str(hen) for hen in max_min_ratio_groups)  # x轴刻度的标识
    center_ratio = 2.5
    plt.xticks(
        henzuobiao_indexes + center_ratio * bar_width, 
        group_labels, 
        fontsize=font_size, 
        fontweight='bold'
    )  # 默认字体大小为10
    plt.yticks(fontsize=font_size, fontweight='bold')
    plt.xlabel(r"Ratio $\lambda$", fontsize=font_size, fontweight='bold')
    plt.ylabel(y_label_name, fontsize=font_size, fontweight='bold')
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

    # plt.xlim(0.9, 6.1)  # 设置x轴的范围
    # plt.ylim(1.5, 16)

    plt.legend()          #显示各曲线的图例
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细
    # plt.legend(loc=4, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
    legend_properties = {
        'weight':'bold',
        'size': font_size
    }
    plt.legend(bbox_to_anchor=(0.5,1.3), labelspacing=0.05, columnspacing=0.2, loc='upper center', ncol=4, prop=legend_properties, frameon=False)

    plt.subplots_adjust(left=0.1, right=0.88)

    plt.tight_layout()
    result_path_prefix = current_dir + "/fig_2_{}".format(y_label_name)
    plt.savefig(result_path_prefix + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()
    pp = PdfPages(result_path_prefix + '.pdf')
    pp.savefig(fig)
    pp.close()