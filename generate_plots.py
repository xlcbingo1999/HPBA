import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill
from utils.plot_operator import add_df_with_min_max, get_mark_color_hatch_marker, get_result_avg_min_max_for_y_label_name

# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def draw_group_plot(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                y_label_name_arr, env_x_label, params, get_policy_map_func):
    current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    path = os.path.join(current_dir, f"{target_pic_name}.csv")
    df = pd.read_csv(path)

    df_with_key = df.set_index(keys_str, drop=False)
    unique_values = df_with_key.index.unique()
    print(unique_values)
    df_with_key = add_df_with_min_max(df_with_key)
    print("---- df.info ----")
    df_with_key.info()

    max_one_line_length = params["max_one_line_length"]
    font_size = params["font_size"]
    line_width = params["line_width"]
    fill_between_alpha = params["fill_between_alpha"]
    bbox_to_anchor = params["bbox_to_anchor"]
    labels_pacing = params["label_spacing"]
    column_spacing = params["column_spacing"]
    ncol = params["ncol"]
    marker_size = params["marker_size"]
    same_distance = params["same_distance"]
    
    colors, _, markers = get_mark_color_hatch_marker()
    for y_label_name in y_label_name_arr:
        results, results_min, results_max = get_result_avg_min_max_for_y_label_name(
            df_with_key, env_policy_groups, env_x_groups, y_label_name
        )

        fig = plt.figure(figsize=(10, 5))
        plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # 去掉上边框
        ax.spines['right'].set_visible(False)  # 去掉右边框

        for policy_index, policy in enumerate(env_policy_groups):
            print(f"policy_index: {policy_index}: policy: {policy}")
            print(f"results[group_index]: {results[policy_index]}")
            if same_distance:
                env_x_groups_str = range(len(env_x_groups))
            else:
                env_x_groups_str = env_x_groups
            plt.plot(
                env_x_groups_str, 
                results[policy_index], 
                marker=markers[policy_index], 
                markersize=marker_size,
                color=colors[policy_index], 
                label=get_policy_map_func(policy), 
                linewidth=line_width
            )
            plt.fill_between(
                env_x_groups_str, 
                results_min[policy_index], 
                results_max[policy_index], 
                color=colors[policy_index], 
                alpha=fill_between_alpha
            )

        group_labels = list(str(hen) for hen in env_x_groups)  # x轴刻度的标识
        plt.xticks(
            env_x_groups_str, 
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
            'size': font_size-2
        }
        plt.legend(bbox_to_anchor=bbox_to_anchor, labelspacing=labels_pacing, columnspacing=column_spacing, loc='upper center', ncol=ncol, prop=legend_properties, frameon=False)

        # plt.subplots_adjust(left=0.1, right=0.88)

        plt.tight_layout()
        result_path_prefix = os.path.join(current_dir, f"{target_pic_name}_{y_label_name}")
        plt.savefig(result_path_prefix + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
        plt.show()
        pp = PdfPages(result_path_prefix + '.pdf')
        pp.savefig(fig)
        pp.close()

def draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                y_label_name_arr, env_x_label, params, get_policy_map_func):
    current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    path = os.path.join(current_dir, f"{target_pic_name}.csv")
    df = pd.read_csv(path)

    df_with_key = df.set_index(keys_str, drop=False)
    unique_values = df_with_key.index.unique()
    print(unique_values)
    df_with_key = add_df_with_min_max(df_with_key)
    print("---- df.info ----")
    df_with_key.info()

    max_one_line_length = params["max_one_line_length"]
    font_size = params["font_size"]
    center_ratio = params["center_ratio"]
    bar_width = params["bar_width"]
    bar_width_ratio = params["bar_width_ratio"]
    bbox_to_anchor = params["bbox_to_anchor"]
    labels_pacing = params["label_spacing"]
    column_spacing = params["column_spacing"]
    ncol = params["ncol"]
    colors, hatchs, _ = get_mark_color_hatch_marker()
    for y_label_name in y_label_name_arr:
        results, results_min, results_max = get_result_avg_min_max_for_y_label_name(
            df_with_key, env_policy_groups, env_x_groups, y_label_name
        )

        fig = plt.figure(figsize=(10, 5))
        plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # 去掉上边框
        ax.spines['right'].set_visible(False)  # 去掉右边框

        henzuobiao_indexes = np.arange(len(env_x_groups)) * bar_width_ratio

        for policy_index, policy in enumerate(env_policy_groups):
            print(f"policy_index: {policy_index}: policy: {policy}")
            print(f"results[group_index]: {results[policy_index]}")
            plt.bar(
                henzuobiao_indexes + policy_index * bar_width, 
                results[policy_index], 
                bar_width, 
                color=colors[policy_index], 
                label=get_policy_map_func(policy),
                hatch=hatchs[policy_index],
                edgecolor="black", 
            )

        group_labels = list(str(hen) for hen in env_x_groups)  # x轴刻度的标识
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
            'size': font_size-2
        }
        plt.legend(bbox_to_anchor=bbox_to_anchor, labelspacing=labels_pacing, columnspacing=column_spacing, loc='upper center', ncol=ncol, prop=legend_properties, frameon=False)

        # plt.subplots_adjust(left=0.1, right=0.88)

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
    env_x_groups = [500, 1000, 1500, 2000, 3000]
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
    def get_fig_1_policy_map(origin_policy):
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
            match = re.match(r"IterativeHISwithOrderPolicy\((?P<iteration_num>\d+)\)", origin_policy)
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
    env_x_label = r"Number of test jobs $n$"
    params = {
        "font_size": 20,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 20,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 4,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
    }
    draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                y_label_name_arr, env_x_label, params, get_fig_1_policy_map)

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
    def get_fig_2_policy_map(origin_policy):
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
            match = re.match(r"IterativeHISwithOrderPolicy\((?P<iteration_num>\d+)\)", origin_policy)
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
    env_x_label = r"Ratio $\lambda$"
    params = {
        "font_size": 20,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 20,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 4,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
    }
    draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                    y_label_name_arr, env_x_label, params, get_fig_2_policy_map)

def draw_fig_6():
    target_pic_name = "fig_6_right"
    keys_str = ["policy", "Max-min Ratio"]
    env_x_groups = [0.025, 0.033, 0.05, 0.1, 0.4, 0.7, 1.0] # 0.025, 0.033, 0.05,
    env_policy_groups = [
        "IterativeHISwithOrderPolicy(0)",
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
    def get_fig_6_policy_map(origin_policy):
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
            match = re.match(r"IterativeHISwithOrderPolicy\((?P<iteration_num>\d+)\)", origin_policy)
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
    env_x_label = r"Ratio $\lambda$"
    params = {
        "font_size": 20,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 20,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 4,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
    }
    draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                    y_label_name_arr, env_x_label, params, get_fig_6_policy_map)

def draw_fig_5():
    target_pic_name = "fig_5_right"
    keys_str = ["policy", "Offline history job num"]
    
    env_policy_groups = [
        "IterativeHISwithOrderPolicy(10)",
        "IterativeHISwithOrderPolicy(20)",
        "IterativeHISwithOrderPolicy(50)",
        "IterativeHISwithOrderPolicy(100)",
        "IterativeHISwithOrderPolicy(250)",
        "IterativeHISwithOrderPolicy(300)",
        "IterativeHISwithOrderPolicy(500)",
        "IterativeHISwithOrderPolicy(1000)",
        "OfflinePolicy"
    ]
    env_x_groups = [
        0, 10, 20, 50, 100, 250, 300, 500, 1000
    ]
    y_label_name_arr = [
        "Significance of all jobs", 
        "Average Significance of allocated jobs",
        "Ratio of Allocated Datablocks"
    ]
    def get_fig_5_policy_map(origin_policy):
        if "IterativeHISwithOrderPolicy" in origin_policy:
            result_policy = "IterativeHIS"
            match = re.match(r"IterativeHISwithOrderPolicy\((?P<iteration_num>\d+)\)", origin_policy)
            if match:
                result_policy = result_policy + "({})".format(match.group("iteration_num"))
        else:
            result_policy = origin_policy
        return result_policy

    env_x_label = r"Number of offline history job"
    params = {
        "font_size": 20,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 20,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
        "marker_size": 10,
        "same_distance": True
    }
    draw_group_plot(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                y_label_name_arr, env_x_label, params, get_fig_5_policy_map)

if __name__ == "__main__":
    # draw_fig_1()
    # draw_fig_2()
    # draw_fig_6()
    draw_fig_5()
    
    # "Number of Allocated Jobs", 
    # "Ratio of Allocated Jobs", 
    # "Significance of all jobs", 
    # "Significance of allocated jobs",
    # "Average Significance of all jobs",
    # "Average Significance of allocated jobs",
    # "Success Datablock Num",
    # "Failed Datablock Num",
    # "Target Datablock Num"