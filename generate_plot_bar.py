import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill
from utils.plot_operator import add_df_with_min_max, get_mark_color_hatch, get_result_avg_min_max_for_y_label_name

# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def get_policy_map(origin_policy):
    result_policy = ""
    if origin_policy == "OfflinePolicy":
        result_policy = "Ground Truth"
    elif "HISwithOrderProVersionPolicy" in origin_policy:
        result_policy = "HIS"
        match = re.match(r"HISwithOrderProVersionPolicy\((?P<history_num>\d+)\)", origin_policy)
        if match:
            result_policy = result_policy + "({})".format(match.group("history_num"))
    elif "IterativeHISwithOrderPolicy" in origin_policy:
        result_policy = "IterativeHIS"
        match = re.match(r"IterativeHIS\((?P<iteration_num>\d+)\)", origin_policy)
        if match:
            result_policy = result_policy + "({})".format(match.group("iteration_num"))
    elif origin_policy == "PBGPolicy":
        result_policy = "PBG"
    elif origin_policy == "PBGMixPolicy": 
        result_policy = "PBGMix"
    elif origin_policy == "SagewithRemainPolicy":
        result_policy = "Sage"
    elif origin_policy == "BestFitwithRemainPolicy":
        result_policy = "BestFit"
    return result_policy

def draw_bar(target_pic_name, keys_str, env_x_groups, env_policy_groups, y_label_name_arr, env_x_label):
    current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    path = os.path.join(current_dir, f"{target_pic_name}.csv")
    df = pd.read_csv(path)

    df_with_key = df.set_index(keys_str, drop=False)
    unique_values = df_with_key.index.unique()
    print(unique_values)
    df_with_key = add_df_with_min_max(df_with_key)
    print("---- df.info ----")
    df_with_key.info()

    max_one_line_length = 20
    colors, hatchs = get_mark_color_hatch()
    for y_label_name in y_label_name_arr:
        results, results_min, results_max = get_result_avg_min_max_for_y_label_name(
            df_with_key, env_x_groups, env_policy_groups, y_label_name
        )

        fig = plt.figure(figsize=(10, 5))
        plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # 去掉上边框
        ax.spines['right'].set_visible(False)  # 去掉右边框

        bar_width = 0.23
        ratio = 2
        font_size = 20
        henzuobiao_indexes = np.arange(len(env_x_groups)) * ratio

        for policy_index, policy in enumerate(env_policy_groups):
            print(f"policy_index: {policy_index}: policy: {policy}")
            print(f"results[group_index]: {results[policy_index]}")
            plt.bar(
                henzuobiao_indexes + policy_index * bar_width, 
                results[policy_index], 
                bar_width, 
                color=colors[policy_index], 
                label=get_policy_map(policy),
                hatch=hatchs[policy_index],
                edgecolor="black", 
            )

        group_labels = list(str(hen) for hen in env_x_groups)  # x轴刻度的标识
        center_ratio = 2.5
        plt.xticks(
            henzuobiao_indexes + center_ratio * bar_width, 
            group_labels, 
            fontsize=font_size, 
            fontweight='bold'
        )
        plt.yticks(fontsize=font_size, fontweight='bold')
        plt.xlabel(env_x_label, fontsize=font_size, fontweight='bold')
        plt.ylabel(fill(y_label_name, max_one_line_length), fontsize=font_size, fontweight='bold')
        if np.mean(results) < 1e-2 or np.mean(results) > 1e2:
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
        plt.legend(bbox_to_anchor=(0.5,1.35), labelspacing=0.05, columnspacing=0.2, loc='upper center', ncol=4, prop=legend_properties, frameon=False)

        plt.subplots_adjust(left=0.1, right=0.88)

        plt.tight_layout()
        result_path_prefix = os.path.join(current_dir, f"{target_pic_name}_{y_label_name}")
        plt.savefig(result_path_prefix + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
        plt.show()
        pp = PdfPages(result_path_prefix + '.pdf')
        pp.savefig(fig)
        pp.close()

def draw_fig_1():
    target_pic_name = "fig_1_right"
    keys_str = ["policy", "Online job num"]
    env_x_groups = [500, 1000, 1500, 2000, 3000, 4000]
    env_policy_groups = [
        "OfflinePolicy",
        "HISwithOrderProVersionPolicy", 
        "IterativeHISwithOrderPolicy(100)", 
        "PBGPolicy",
        "PBGMixPolicy", 
        "SagewithRemainPolicy",
        "BestFitwithRemainPolicy"
    ]
    y_label_name_arr = [
        "Significance of all jobs", 
        "Average Significance of allocated jobs",
        "Ratio of Allocated Datablocks"
    ] 
    env_x_label = r"Number of test jobs $n$"
    draw_bar(target_pic_name, keys_str, env_x_groups, env_policy_groups, y_label_name_arr, env_x_label)

def draw_fig_2():
    target_pic_name = "fig_2_right"
    keys_str = ["policy", "Max-min Ratio"]
    env_x_groups = [0.1, 0.4, 0.7, 1.0]
    env_policy_groups = [
        "OfflinePolicy",
        "HISwithOrderProVersionPolicy", 
        "IterativeHISwithOrderPolicy(100)", 
        "PBGPolicy",
        "PBGMixPolicy", 
        "SagewithRemainPolicy",
        "BestFitwithRemainPolicy"
    ]
    y_label_name_arr = [
        "Significance of all jobs", 
        "Average Significance of allocated jobs",
        "Ratio of Allocated Datablocks"
    ]
    env_x_label = r"Ratio $\lambda$"
    draw_bar(target_pic_name, keys_str, env_x_groups, env_policy_groups, y_label_name_arr, env_x_label)

def draw_fig_6():
    target_pic_name = "fig_6_right"
    keys_str = ["policy", "Max-min Ratio"]
    env_x_groups = [0.025, 0.033, 0.05, 0.1, 0.4, 0.7, 1.0] # 0.025, 0.033, 0.05,
    env_policy_groups = [
        "HISwithOrderProVersionPolicy(0)",
        "HISwithOrderProVersionPolicy(200)",
        "HISwithOrderProVersionPolicy(400)",
        "HISwithOrderProVersionPolicy(600)",
        "HISwithOrderProVersionPolicy(800)",
        "HISwithOrderProVersionPolicy(1000)",
        "OfflinePolicy",
    ]
    y_label_name_arr = [
        "Significance of all jobs", 
        "Average Significance of allocated jobs",
        "Ratio of Allocated Datablocks"
    ]
    env_x_label = r"Ratio $\lambda$"
    draw_bar(target_pic_name, keys_str, env_x_groups, env_policy_groups, y_label_name_arr, env_x_label)

if __name__ == "__main__":
    draw_fig_1()
    draw_fig_2()
    draw_fig_6()
    
    # "Number of Allocated Jobs", 
    # "Ratio of Allocated Jobs", 
    # "Significance of all jobs", 
    # "Significance of allocated jobs",
    # "Average Significance of all jobs",
    # "Average Significance of allocated jobs",
    # "Success Datablock Num",
    # "Failed Datablock Num",
    # "Target Datablock Num"