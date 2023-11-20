import numpy as np
import pandas as pd
import re
import os
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill
from utils.plot_operator import add_df_with_min_max, get_result_avg_min_max_for_y_label_name

plt.rcParams['font.family'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['pdf.fonttype'] = 42

def xlsx_2_csv(file_name):
    sheet_name_2_result_file_name = {
        "Sheet1": file_name,
    }
    # 定义输入的 Excel 文件路径
    plots_path = "/home/netlab/DL_lab/opacus_testbed/plots"
    input_excel_file = os.path.join(plots_path, '{}.xlsx'.format(file_name)) 

    # 定义输出的 CSV 文件路径
    for sheet_name, result_file_name in sheet_name_2_result_file_name.items():
        output_csv_file = os.path.join(plots_path, '{}.csv'.format(result_file_name))

        # 读取 Excel 文件中的数据
        data_frame = pd.read_excel(input_excel_file, sheet_name=sheet_name)
        data_frame.info()

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

def from_y_label_name_2_add_columns_keys_2_need_max_map(y_label_name_arr):
    add_columns_keys_2_need_max_map = {} 
    # "Significance of all queries", "Sum of Delta Accuracy"
    for y_label_name in y_label_name_arr:
        if y_label_name == "Number of Allocated Jobs":
            add_columns_keys_2_need_max_map["Success num"] = True
        elif y_label_name == "Number of Failed Jobs":
            add_columns_keys_2_need_max_map["Failed num"] = True
        elif y_label_name == "Ratio of Allocated Jobs":
            add_columns_keys_2_need_max_map["Success num"] = True
        elif y_label_name == "Ratio of Allocated Datablocks":
            add_columns_keys_2_need_max_map["Success Datablock Num"] = True
            add_columns_keys_2_need_max_map["Target Datablock Num"] = True
        elif y_label_name == "Significance of all queries" or y_label_name == "Total Significance Score":
            add_columns_keys_2_need_max_map["Mean Significance All"] = True
        elif y_label_name == "Significance of allocated queries":
            add_columns_keys_2_need_max_map["Mean Significance Success"] = True
            add_columns_keys_2_need_max_map["Success num"] = True
        elif y_label_name == "Sum of Delta Train Accuracy":
            add_columns_keys_2_need_max_map["Train Accuracy All"] = True
        elif y_label_name == "Sum of Delta Test Accuracy"  or y_label_name == "Total Test Accuracy Improvement (%)":
            add_columns_keys_2_need_max_map["Test Accuracy All"] = True
        elif y_label_name == "Sum of Delta Train Loss":
            add_columns_keys_2_need_max_map["Train Loss All"] = False
        elif y_label_name == "Sum of Delta Test Loss":
            add_columns_keys_2_need_max_map["Test Loss All"] = False
        elif y_label_name == "Epsilon_Real_All_Block":
            add_columns_keys_2_need_max_map[y_label_name] = False
        elif (y_label_name == "Average Decision Time Consumption (s)"):
            add_columns_keys_2_need_max_map["Decision_Duration"] = False
        elif (y_label_name == "Epsilon_Real_All_Block"
            or y_label_name == "Significance_Epsilon_Ratio" 
            or y_label_name == "Test_Loss_Epsilon_Ratio"
            or y_label_name == "Test_Accuracy_Epsilon_Ratio"
        ):
            add_columns_keys_2_need_max_map[y_label_name] = True
    return add_columns_keys_2_need_max_map

def draw_plot_worker(fill_between_flag, results, results_min, results_max,
                    env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name, env_x_label, get_policy_map_func,
                    params,
                    current_dir, target_pic_name,
                    get_mark_color_hatch_marker_func):
    max_one_line_length = params["max_one_line_length"]
    font_size = params["font_size"]
    legend_font_size = params["legend_font_size"]
    line_width = params["line_width"]
    fill_between_alpha = params["fill_between_alpha"]
    bbox_to_anchor = params["bbox_to_anchor"]
    labels_pacing = params["label_spacing"]
    column_spacing = params["column_spacing"]
    ncol = params["ncol"]
    marker_size = params["marker_size"]
    same_distance = params["same_distance"]
    figsize = params["figsize"] if "figsize" in params else None
    max_x_label_show_list = params["max_x_label_show_list"] if "max_x_label_show_list" in params else None
    ylim = params["ylim"] if "ylim" in params else None
    order = params["order"] if "order" in params else None

    colors, _, markers = get_mark_color_hatch_marker_func()

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    for policy_index, policy in enumerate(env_policy_groups):
        is_default = policy_index in env_policy_default_indexes
        # print(f"policy_index: {policy_index}: policy: {policy}, label: {get_policy_map_func(policy, is_default)}")
        # print(f"results[group_index]: {results[policy_index]}")
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
            label=get_policy_map_func(policy, is_default), 
            linewidth=line_width
        )
        if fill_between_flag:
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
    # if len(env_x_label) > max_one_line_length:
    #     plt.xlabel(fill(env_x_label, max_one_line_length), fontsize=font_size-2, fontweight='bold')
    # else:
    
    if len(y_label_name) > max_one_line_length:
        plt.xlabel(env_x_label, fontsize=font_size-2, fontweight='bold')
        plt.ylabel(fill(y_label_name, max_one_line_length), fontsize=font_size-2, fontweight='bold')
    else:
        plt.xlabel(env_x_label, fontsize=font_size, fontweight='bold')
        plt.ylabel(y_label_name, fontsize=font_size, fontweight='bold')
    if np.mean(results) < 1e-3 or np.mean(results) > 1e3:
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    # plt.xlim(0.9, 6.1)  # 设置x轴的范围
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    # plt.ylim(1.5, 16)
    if max_x_label_show_list is not None:
        ax.xaxis.set_major_locator(ticker.FixedLocator(max_x_label_show_list)) 

    plt.legend()          #显示各曲线的图例
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细
    # plt.legend(loc=4, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
    legend_properties = {
        'weight':'bold',
        'size': legend_font_size
    }
    if order is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], bbox_to_anchor=bbox_to_anchor, labelspacing=labels_pacing, columnspacing=column_spacing, loc='upper center', ncol=ncol, prop=legend_properties, frameon=False)
    else:
        plt.legend(bbox_to_anchor=bbox_to_anchor, labelspacing=labels_pacing, columnspacing=column_spacing, loc='upper center', ncol=ncol, prop=legend_properties, frameon=False)
    
    # plt.subplots_adjust(left=0.1, right=0.88)

    plt.tight_layout()
    result_path_prefix = os.path.join(current_dir, f"{target_pic_name}_{y_label_name}")
    result_path_prefix = result_path_prefix.replace(" ", "_")
    if "(%)" in result_path_prefix:
        result_path_prefix = result_path_prefix.replace("(%)", "")
    if "(s)" in result_path_prefix:
        result_path_prefix = result_path_prefix.replace("(s)", "")
    plt.savefig(result_path_prefix + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()
    pp = PdfPages(result_path_prefix + '.pdf')
    pp.savefig(fig)
    pp.close()

def draw_bar_worker(results, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name, env_x_label, get_policy_map_func,
                    params,
                    current_dir, target_pic_name,
                    get_mark_color_hatch_marker_func):
    max_one_line_length = params["max_one_line_length"]
    font_size = params["font_size"]
    legend_font_size = params["legend_font_size"]
    center_ratio = params["center_ratio"]
    bar_width = params["bar_width"]
    bar_width_ratio = params["bar_width_ratio"]
    bbox_to_anchor = params["bbox_to_anchor"]
    labels_pacing = params["label_spacing"]
    column_spacing = params["column_spacing"]
    ncol = params["ncol"]
    figsize = params["figsize"] if "figsize" in params else None
    ylim = params["ylim"] if "ylim" in params else None
    order = params["order"] if "order" in params else None

    colors, hatchs, _ = get_mark_color_hatch_marker_func()

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    plt.grid(linestyle="--", axis='y', alpha=0.5)  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    henzuobiao_indexes = np.arange(len(env_x_groups)) * bar_width_ratio

    for policy_index, policy in enumerate(env_policy_groups):
        is_default = policy_index in env_policy_default_indexes
        print(f"policy_index: {policy_index}: policy: {policy}")
        print(f"results[group_index]: {results[policy_index]}")
        plt.bar(
            henzuobiao_indexes + policy_index * bar_width, 
            results[policy_index], 
            bar_width, 
            color=colors[policy_index], 
            label=get_policy_map_func(policy, is_default),
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
    # if len(env_x_label) > max_one_line_length:
    #     plt.xlabel(fill(env_x_label, max_one_line_length), fontsize=font_size-2, fontweight='bold')
    # else:
    
    if len(y_label_name) > max_one_line_length:
        plt.xlabel(env_x_label, fontsize=font_size-2, fontweight='bold')
        plt.ylabel(fill(y_label_name, max_one_line_length), fontsize=font_size-2, fontweight='bold')
    else:
        plt.xlabel(env_x_label, fontsize=font_size, fontweight='bold')
        plt.ylabel(y_label_name, fontsize=font_size, fontweight='bold')
    if np.mean(results) < 1e-3 or np.mean(results) > 1e3:
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

    # plt.xlim(0.9, 6.1)  # 设置x轴的范围
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend()          #显示各曲线的图例
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # fontweight='bold' 设置图例字体的大小和粗细
    # plt.legend(loc=4, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
    legend_properties = {
        'weight':'bold',
        'size': legend_font_size
    }
    if order is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], bbox_to_anchor=bbox_to_anchor, labelspacing=labels_pacing, columnspacing=column_spacing, loc='upper center', ncol=ncol, prop=legend_properties, frameon=False)
    else:
        plt.legend(bbox_to_anchor=bbox_to_anchor, labelspacing=labels_pacing, columnspacing=column_spacing, loc='upper center', ncol=ncol, prop=legend_properties, frameon=False)

    # plt.subplots_adjust(left=0.1, right=0.88)

    plt.tight_layout()
    result_path_prefix = os.path.join(current_dir, f"{target_pic_name}_{y_label_name}")
    result_path_prefix = result_path_prefix.replace(" ", "_")
    if "(%)" in result_path_prefix:
        result_path_prefix = result_path_prefix.replace("(%)", "")
    if "(s)" in result_path_prefix:
        result_path_prefix = result_path_prefix.replace("(s)", "")
    plt.savefig(result_path_prefix + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()
    pp = PdfPages(result_path_prefix + '.pdf')
    pp.savefig(fig)
    pp.close()


def get_result_and_draw_group_plot(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                y_label_name_arr, env_x_label, params, fill_between_flag, get_policy_map_func, get_mark_color_hatch_marker_func):
    current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    path = os.path.join(current_dir, f"{target_pic_name}.csv")
    df = pd.read_csv(path)

    df_with_key = df.set_index(keys_str, drop=False)
    unique_values = df_with_key.index.unique()
    print(unique_values)

    add_columns_keys_2_need_max_map = from_y_label_name_2_add_columns_keys_2_need_max_map(y_label_name_arr)
    df_with_key = add_df_with_min_max(df_with_key, add_columns_keys_2_need_max_map)
    print("---- df.info ----")
    df_with_key.info()
    # 临时保存一下!
    temp_pa = os.path.join(current_dir, f"{target_pic_name}_df_with_key.xlsx")
    df_with_key.to_excel(temp_pa, index=False)

    
    for y_label_name in y_label_name_arr:
        results, results_min, results_max = get_result_avg_min_max_for_y_label_name(
            df_with_key, env_policy_groups, env_x_groups, y_label_name
        )
        draw_plot_worker(fill_between_flag, results, results_min, results_max,
                        env_policy_groups, env_policy_default_indexes, env_x_groups, 
                        y_label_name, env_x_label, get_policy_map_func, 
                        params,
                        current_dir, target_pic_name,
                        get_mark_color_hatch_marker_func)
        

def get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups,
                y_label_name_arr, env_x_label, params, get_policy_map_func, get_mark_color_hatch_marker_func):
    current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    path = os.path.join(current_dir, f"{target_pic_name}.csv")
    df = pd.read_csv(path)

    df_with_key = df.set_index(keys_str, drop=False)
    unique_values = df_with_key.index.unique()
    print(unique_values)
    
    add_columns_keys_2_need_max_map = from_y_label_name_2_add_columns_keys_2_need_max_map(y_label_name_arr)
    df_with_key = add_df_with_min_max(df_with_key, add_columns_keys_2_need_max_map)
    print("---- df.info ----")
    df_with_key.info()
    # 临时保存一下!
    temp_pa = os.path.join(current_dir, f"{target_pic_name}_df_with_key.xlsx")
    df_with_key.to_excel(temp_pa, index=False)

    for y_label_name in y_label_name_arr:
        results, results_min, results_max = get_result_avg_min_max_for_y_label_name(
            df_with_key, env_policy_groups, env_x_groups, y_label_name
        )
        draw_bar_worker(results, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                        y_label_name, env_x_label, get_policy_map_func, 
                        params,
                        current_dir, target_pic_name,
                        get_mark_color_hatch_marker_func)
    
def draw_cr():
    def get_cr_v2(lamb, h_n_ratio):
        if (1 - lamb) <= 0:
            return -1000000
        else:
            return (2 - lamb - (1 + h_n_ratio) * math.log(1 + 1/h_n_ratio)) / (1 - lamb)

    
    env_x_groups = [i * 5 / 100 for i in range(0, 100, 10) if i > 0] # h/s
    env_policy_groups = [i / 100 for i in range(0, 100, 20) if i > 0] # lambda
    
    args_product_list = [d for d in itertools.product(env_policy_groups, env_x_groups)]
    results = []
    for lamb in env_policy_groups:
        temp_result = []
        for h_n_ratio in env_x_groups:
            res = max(0.0, get_cr_v2(lamb, h_n_ratio)) 
            temp_result.append(res)
        results.append(temp_result)

    
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 30,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
        "marker_size": 10,
        "same_distance": True,
        # "figsize": (8, 6),
        # "max_x_label_show_list": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    }
    y_label_name = "Competition Ratio"
    env_x_label = r"$\frac{h}{n}$"
    def get_policy_map_func(origin_policy):
        return r"$\lambda = {}$".format(origin_policy)
    current_dir = "/home/netlab/DL_lab/opacus_testbed/plots"
    target_pic_name = "cr"
    draw_plot_worker(False, results, None, None,
                    env_policy_groups, env_x_groups, 
                    y_label_name, env_x_label, get_policy_map_func, 
                    params,
                    current_dir, target_pic_name)
    
def draw_F1():
    target_pic_name = "testbed_F1_history0_historyinf"
    xlsx_2_csv(target_pic_name)
    keys_str = ["policy", "Datablock num"]
    env_x_groups = [40, 60, 80, 100] # Datablock num
    env_policy_groups = [
        "HISwithOrderProVersionPolicy(0,800)",
        "IterativeHISwithOrderProVersionPolicy(100,800)",
        "HISwithOrderProVersionPolicy(0,1)",
        "IterativeHISwithOrderProVersionPolicy(1,1)",
        "BestFitwithRemainPolicy",
        "SagewithRemainPolicy",
        "PBGPolicy", 
        "OfflinePolicy"
    ]
    env_policy_default_indexes = [0, 1]
    def get_mark_color_hatch_marker():
        colors =["#ffd6a5", "#fdffb6",  "#caffbf", "#9bf6ff",  "#bdb2ff", "#ffc6ff", "#a0c4ff",
                "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...', '////',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_F1_policy_map(origin_policy, is_default):
        result_policy = ""
        if origin_policy == "OfflinePolicy":
            result_policy = "Optimal"
        elif "IterativeHISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "S-HPBA"
            if is_default:
                result_policy = result_policy + "(default)"
                # result_policy = result_policy + r"(w/ $\mathcal{H}$)"
            else:
                num_match = re.match(r"IterativeHISwithOrderProVersionPolicy\((?P<batch_size>\d+),(?P<history_num>\d+)\)", origin_policy)
                if num_match:
                    result_policy = result_policy + r"($\psi=$" + "{},".format(int(num_match.group("batch_size"))) + r"$H$=" + "{})".format(int(num_match.group("history_num")))
        elif "HISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "HPBA"
            if is_default:
                result_policy = result_policy + "(default)"
                # result_policy = result_policy + r"(w/ $\mathcal{H}$)"
            else:
                num_match = re.match(r"HISwithOrderProVersionPolicy\((?P<batch_size>\d+),(?P<history_num>\d+)\)", origin_policy)
                if num_match:
                    result_policy = result_policy + r"($H$=" + "{})".format(int(num_match.group("history_num")))
        elif origin_policy == "PBGPolicy":
            result_policy = "Sig"
        elif origin_policy == "PBGMixPolicy": 
            result_policy = "PBGMix"
        elif origin_policy == "SagewithRemainPolicy":
            result_policy = "Sage"
        elif origin_policy == "BestFitwithRemainPolicy":
            result_policy = "BestFit"
        return result_policy
    env_x_label = r"Number of private datablocks ($|\mathcal{D}|$)"
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.2,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 28,
        "bbox_to_anchor": (0.45,1.32),
        "label_spacing": 0.05,
        "column_spacing": 0.1,
        "ncol": 3,
        "center_ratio": 3.5,
        "bar_width_ratio": 2,
        "order": [0, 3, 6, 1, 4, 7, 2, 5],
    }
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name_arr, env_x_label, params, get_F1_policy_map, get_mark_color_hatch_marker)

def draw_Q2():
    target_pic_name = "testbed_Q2"
    xlsx_2_csv(target_pic_name)
    keys_str = ["policy", "Datablock num"]
    env_x_groups = [20, 40, 60, 80, 100] # Datablock num 0, 50, 100, 150, 250, 300
    env_policy_groups = [
        "HISwithOrderProVersionPolicy(0)",  # 0.0
        # "HISwithOrderProVersionPolicy(25)", # 0.4
        # "HISwithOrderProVersionPolicy(50)", # 0.4
        # "HISwithOrderProVersionPolicy(100)", # 0.8
        # "HISwithOrderProVersionPolicy(150)", # 1.2
        # "HISwithOrderProVersionPolicy(200)", # 1.6
        # "HISwithOrderProVersionPolicy(300)", # 2.0
        "HISwithOrderProVersionPolicy(400)", # 2.0
        "HISwithOrderProVersionPolicy(800)", # 2.0
        "HISwithOrderProVersionPolicy(1200)", # 2.0
        "HISwithOrderProVersionPolicy(1600)", # 2.0
        # "HISwithOrderProVersionPolicy(2000)", # 2.0
        "OfflinePolicy",
    ]
    env_policy_default_indexes = [2]
    def get_mark_color_hatch_marker():
        colors =["#ffd6a5", "#fdffb6",  "#caffbf", "#9bf6ff",  "#bdb2ff", "#ffc6ff", "#936639",
                "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...', 'x',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_Q2_policy_map(origin_policy, is_default):
        result_policy = ""
        if "HISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "HPBA"
            match = re.match(r"HISwithOrderProVersionPolicy\((?P<history_num>\d+)\)", origin_policy)
            if is_default:
                result_policy = result_policy + "(default)"
            elif match:
                result_policy = result_policy + r"($H$=" + "{})".format(int(match.group("history_num")))
        elif "OfflinePolicy" in origin_policy:
            result_policy = "Optimal"
        return result_policy
    env_x_label = r"Number of private datablocks ($|\mathcal{D}|$)"
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.25,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 28,
        "bbox_to_anchor": (0.45,1.25),
        "label_spacing": 0.05,
        "column_spacing": 0.1,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
    }
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name_arr, env_x_label, params, get_Q2_policy_map, get_mark_color_hatch_marker)

def draw_Q5_plot():
    # SAHIS
    target_pic_name = "testbed_Q5_plot"
    xlsx_2_csv(target_pic_name)
    keys_str = ["Datablock num", "Offline history job num"]
    env_x_groups = [0, 25, 50, 100, 150, 200, 300, 800, 1600, 2000] # Datablock num 0, 50, 100, 150, 250, 300
    env_policy_groups = [
        40
    ]
    def get_mark_color_hatch_marker():
        colors =["#ffd6a5", "#fdffb6",  "#caffbf", "#9bf6ff",  "#bdb2ff", "#ffc6ff", "#936639",
                "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...', 'x',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_Q2_plot_policy_map(origin_policy):
        result_policy = r"$|\mathcal{D}|$=" + f"{origin_policy}"
        return result_policy
    env_x_label = r"Number of offline history records ($|\mathcal{H}|$)"
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 30,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
        "marker_size": 10,
        "same_distance": True,
    }
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    get_result_and_draw_group_plot(target_pic_name, keys_str, env_policy_groups, env_x_groups, 
                    y_label_name_arr, env_x_label, params, False, get_Q2_plot_policy_map, get_mark_color_hatch_marker)


def draw_F5():
    target_pic_name = "testbed_F5"
    xlsx_2_csv(target_pic_name)
    keys_str = ["policy", "Datablock num"]
    env_x_groups = [40, 60, 80, 100] # Datablock num 1, 10, 50, 100, 150, 200
    env_policy_groups = [
        "IterativeHISwithOrderProVersionPolicy(1,800)", 
        # "IterativeHISwithOrderProVersionPolicy(10)", 
        "IterativeHISwithOrderProVersionPolicy(50,800)", 
        "IterativeHISwithOrderProVersionPolicy(100,800)", 
        # "IterativeHISwithOrderProVersionPolicy(150)", 
        "IterativeHISwithOrderProVersionPolicy(200,800)", 
        "IterativeHISwithOrderProVersionPolicy(400,800)", 
        # "IterativeHISwithOrderProVersionPolicy(1000)", 
        "HISwithOrderProVersionPolicy(0,800)",
        "OfflinePolicy",
    ]
    env_policy_default_indexes = []
    def get_mark_color_hatch_marker():
        colors =["#ffd6a5", "#fdffb6",  "#caffbf", "#9bf6ff",  "#bdb2ff", "#ffc6ff",
                "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_F5_policy_map(origin_policy, is_default):
        result_policy = ""
        if "IterativeHISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "S-HPBA"
            match = re.match(r"IterativeHISwithOrderProVersionPolicy\((?P<batch_size>\d+),(?P<history_num>\d+)\)", origin_policy)
            if is_default:
                result_policy = result_policy + "(default)"
            elif match:
                result_policy = result_policy + r"($\psi$=" + "{})".format(int(match.group("batch_size")))
        elif "HISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "HPBA"
        elif "OfflinePolicy" in origin_policy:
            result_policy = "Optimal"
        return result_policy
    env_x_label = r"Number of private datablocks ($|\mathcal{D}|$)"
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 28,
        "bbox_to_anchor": (0.45,1.32),
        "label_spacing": 0.05,
        "column_spacing": 0.1,
        "ncol": 3,
        "center_ratio": 3.0,
        "bar_width_ratio": 2,
        "order": [0, 3, 6, 1, 4, 2, 5]
    }
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name_arr, env_x_label, params, get_F5_policy_map, get_mark_color_hatch_marker)

    env_policy_groups.remove("OfflinePolicy")
    print(f"time_draw_env_policy_groups: {env_policy_groups}")
    time_draw_y_label_name_arr = [
        "Average Decision Time Consumption (s)"
    ]
    params["order"] = [0, 3, 1, 4, 2, 5]
    get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    time_draw_y_label_name_arr, env_x_label, params, get_F5_policy_map, get_mark_color_hatch_marker)

def draw_F4():
    target_pic_name = "testbed_F4_history0_historyinf"
    xlsx_2_csv(target_pic_name)
    keys_str = ["policy", "lambda"]
    env_x_groups = [0.1, 0.2, 0.4, 0.8] # 0.05, Datablock num
    env_policy_groups = [
        "HISwithOrderProVersionPolicy",
        "IterativeHISwithOrderProVersionPolicy",
        "BestFitwithRemainPolicy",
        "SagewithRemainPolicy",
        "PBGPolicy", 
        "OfflinePolicy"
    ]
    env_policy_default_indexes = []
    def get_mark_color_hatch_marker():
        # plot
        colors =["#0a9396", "#9b2226", "#005f73", "#936639", "#ca6702", "#94d2bd", "#0a9396", "#9b2226", "#005f73",
                "#ffadad"]
        # bar
        # colors =["#ffd6a5", "#fdffb6",  "#caffbf", "#9bf6ff",  "#bdb2ff", "#ffc6ff", "#a0c4ff",
        #         "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...', '////',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_F4_policy_map(origin_policy, is_default):
        result_policy = ""
        if origin_policy == "OfflinePolicy":
            result_policy = "Optimal"
        elif "IterativeHISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "S-HPBA"
            # match = re.match(r"IterativeHISwithOrderProVersionPolicy\((?P<batch_size>\d+),(?P<history_num>\d+)\)", origin_policy)
            # if is_default:
            #     result_policy = result_policy + r"(default)"
            # elif match:
            #     result_policy = result_policy + r"($\psi=$" + "{},".format(int(match.group("batch_size"))) + r"$H$=" + "{})".format(int(match.group("history_num")))
        elif "HISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "HPBA"
            # match = re.match(r"HISwithOrderProVersionPolicy\((?P<batch_size>\d+),(?P<history_num>\d+)\)", origin_policy)
            # if is_default:
            #     result_policy = result_policy + r"(default)"
            # else:
            #     result_policy = result_policy + r"($H$=" + "{})".format(int(match.group("history_num")))
        elif origin_policy == "PBGPolicy":
            result_policy = "Sig"
        elif origin_policy == "PBGMixPolicy": 
            result_policy = "PBGMix"
        elif origin_policy == "SagewithRemainPolicy":
            result_policy = "Sage"
        elif origin_policy == "BestFitwithRemainPolicy":
            result_policy = "BestFit"
        return result_policy
    env_x_label = r"$\lambda$" # $\frac{r_{i}}{\epsilon_{d}^{G}}$
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 30,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
        "marker_size": 10,
        "same_distance": True,
        "order": [0, 3, 1, 4, 2, 5]
    }
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    get_result_and_draw_group_plot(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                                y_label_name_arr, env_x_label, params, False, get_F4_policy_map, get_mark_color_hatch_marker)

    # bar
    # params = {
    #     "font_size": 20,
    #     "legend_font_size": 15,
    #     "line_width": 1.5,
    #     "bar_width": 0.25,
    #     "fill_between_alpha": 0.5,
    #     "max_one_line_length": 28,
    #     "bbox_to_anchor": (0.45,1.25),
    #     "label_spacing": 0.05,
    #     "column_spacing": 0.1,
    #     "ncol": 3,
    #     "center_ratio": 2.5,
    #     "bar_width_ratio": 2,
    # }
    # y_label_name_arr = [
    #     "Total Significance Score", 
    #     "Total Test Accuracy Improvement (%)",
    # ]
    # get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
    #                 y_label_name_arr, env_x_label, params, get_F4_policy_map, get_mark_color_hatch_marker)
    



def draw_Q5():
    target_pic_name = "testbed_Q5"
    xlsx_2_csv(target_pic_name)
    keys_str = ["policy", "Datablock num"]
    env_x_groups = [20, 40, 60, 80, 100] # Datablock num 0, 50, 100, 150, 250, 300
    env_policy_groups = [
        "IterativeHISwithOrderProVersionPolicy(100,0)",  # 0.0
        "IterativeHISwithOrderProVersionPolicy(100,400)", # 1.6
        "IterativeHISwithOrderProVersionPolicy(100,800)", # 2.0
        "IterativeHISwithOrderProVersionPolicy(100,1200)", # 2.0
        "IterativeHISwithOrderProVersionPolicy(100,1600)", # 2.0
        "OfflinePolicy",
    ]
    env_policy_default_indexes = [2]
    def get_mark_color_hatch_marker():
        colors =["#ffd6a5", "#fdffb6",  "#caffbf", "#9bf6ff",  "#bdb2ff", "#ffc6ff", "#936639",
                "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...', 'x',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_Q5_policy_map(origin_policy, is_default):
        result_policy = ""
        if "IterativeHISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "S-HPBA"
            match = re.match(r"IterativeHISwithOrderProVersionPolicy\((?P<batch_size>\d+),(?P<history_num>\d+)\)", origin_policy)
            if is_default:
                result_policy = result_policy + "(default)"
            elif match:
                result_policy = result_policy + r"($H$=" + "{})".format(int(match.group("history_num")))
        elif "OfflinePolicy" in origin_policy:
            result_policy = "Optimal"
        return result_policy
    env_x_label = r"Number of private datablocks ($|\mathcal{D}|$)"
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.25,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 28,
        "bbox_to_anchor": (0.45,1.25),
        "label_spacing": 0.05,
        "column_spacing": 0.1,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
    }
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name_arr, env_x_label, params, get_Q5_policy_map, get_mark_color_hatch_marker)


def draw_F3():
    target_pic_name = "testbed_F3"
    xlsx_2_csv(target_pic_name)
    keys_str = ["policy", "Offline history job num"]
    env_x_groups = [1, 50, 100, 200, 400, 800, 1200, 1600] # Datablock num 0, 50, 100, 150, 250, 300
    env_policy_groups = [
        "HISwithOrderProVersionPolicy",  # 0.0
        "IterativeHISwithOrderProVersionPolicy",  # 0.0
        "BestFitwithRemainPolicy",
        "SagewithRemainPolicy",
        "PBGPolicy", 
        "OfflinePolicy"
    ]
    env_policy_default_indexes = []
    def get_mark_color_hatch_marker():
        # "#0a9396", "#9b2226", "#005f73",
        colors =["#0a9396", "#9b2226", "#005f73", "#936639", "#ca6702", "#94d2bd", "#0a9396", "#9b2226", "#005f73",
                "#ffadad"]
        hatchs = ['-', '*', '/', 'o', '\\\\',  '...', 'x', '////', '-', '*', '/', 'o',
                '']
        markers = ['x', 'o', 'v', '^', '<', '>', 'P', 's']
        return colors, hatchs, markers
    def get_F3_policy_map(origin_policy, is_default):
        result_policy = ""
        if "IterativeHISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "S-HPBA"
        elif "HISwithOrderProVersionPolicy" in origin_policy:
            result_policy = "HPBA"
        elif origin_policy == "PBGPolicy":
            result_policy = "Sig"
        elif origin_policy == "PBGMixPolicy": 
            result_policy = "PBGMix"
        elif origin_policy == "SagewithRemainPolicy":
            result_policy = "Sage"
        elif origin_policy == "BestFitwithRemainPolicy":
            result_policy = "BestFit"
        elif "OfflinePolicy" in origin_policy:
            result_policy = "Optimal"
        return result_policy
    env_x_label = r"Number of offline historical records ($H$)"
    
    y_label_name_arr = [
        "Total Significance Score", 
        "Total Test Accuracy Improvement (%)",
    ]
    
    # bar
    # params = {
    #     "font_size": 20,
    #     "legend_font_size": 15,
    #     "line_width": 1.5,
    #     "bar_width": 0.5,
    #     "fill_between_alpha": 0.5,
    #     "max_one_line_length": 28,
    #     "bbox_to_anchor": (0.45,1.25),
    #     "label_spacing": 0.05,
    #     "column_spacing": 0.1,
    #     "ncol": 3,
    #     "center_ratio": 0.5,
    #     "bar_width_ratio": 2,
    # }
    # params["ylim"] = [80, 125]
    # def add_threshold():
    #     return 
    #     label_2_y_line = {
    #         "Sig": 82.71586834016392,
    #         "Sage": 81.53955078124999,
    #         "BestFit": 81.24527727971311,
    #         "Optimal": 121.37236648309425
    #     }
    #     for label, y_line in label_2_y_line.items():
    #         plt.axhline(y=y_line, color='red', linestyle='dashed')
    #         plt.annotate(label, xy=(1, y_line), xytext=(2, y_line + 5),
    #                     arrowprops=dict(arrowstyle='->'))
    # get_result_and_draw_group_bar(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
    #                 y_label_name_arr, env_x_label, params, get_F3_policy_map, get_mark_color_hatch_marker, add_threshold)

    # plot
    params = {
        "font_size": 20,
        "legend_font_size": 15,
        "line_width": 1.5,
        "bar_width": 0.23,
        "fill_between_alpha": 0.5,
        "max_one_line_length": 30,
        "bbox_to_anchor": (0.5,1.35),
        "label_spacing": 0.05,
        "column_spacing": 0.2,
        "ncol": 3,
        "center_ratio": 2.5,
        "bar_width_ratio": 2,
        "marker_size": 10,
        "same_distance": True,
        "order": [0, 3, 1, 4, 2, 5]
    }
    params["ylim"] = [80, 125]
    get_result_and_draw_group_plot(target_pic_name, keys_str, env_policy_groups, env_policy_default_indexes, env_x_groups, 
                    y_label_name_arr, env_x_label, params, False, get_F3_policy_map, get_mark_color_hatch_marker)


'''
# 折线图
params = {
    "font_size": 20,
    "legend_font_size": 15,
    "line_width": 1.5,
    "bar_width": 0.23,
    "fill_between_alpha": 0.5,
    "max_one_line_length": 30,
    "bbox_to_anchor": (0.5,1.35),
    "label_spacing": 0.05,
    "column_spacing": 0.2,
    "ncol": 3,
    "center_ratio": 2.5,
    "bar_width_ratio": 2,
    "marker_size": 10,
    "same_distance": True,
}

# bar
params = {
    "font_size": 20,
    "legend_font_size": 15,
    "line_width": 1.5,
    "bar_width": 0.25,
    "fill_between_alpha": 0.5,
    "max_one_line_length": 28,
    "bbox_to_anchor": (0.45,1.25),
    "label_spacing": 0.05,
    "column_spacing": 0.1,
    "ncol": 3,
    "center_ratio": 2.5,
    "bar_width_ratio": 2,
}
'''

if __name__ == "__main__":
    # draw_cr()
    # draw_Q2()
    # draw_Q5()
    draw_F1()
    draw_F3()
    draw_F4()
    draw_F5()
