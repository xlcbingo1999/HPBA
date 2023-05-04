# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

henzuobiaos = np.array([1000, 2000, 3000, 4000])

zhongzuobiao_type = "significance_success" # "significance_all" "significance_success"

iterative_his_gamma_0_0 = np.array([460, 646, 641, 639])
his_gamma_0_0 = np.array([308, 286, 300, 390])
pbg = np.array([381, 292, 350, 380])
pbg_mix = np.array([434, 384, 460, 500])
sage = np.array([383, 298, 350, 380])
best_fit = np.array([376, 287, 350, 380])

ratio_arr_iterative_his_gamma_0_0 = [item/iterative_his_gamma_0_0[0] for item in iterative_his_gamma_0_0]
ratio_arr_his_gamma_0_0 = [item/his_gamma_0_0[0] for item in his_gamma_0_0]
ratio_arr_pbg = [item/pbg[0] for item in pbg]
ratio_arr_pbg_mix = [item/pbg_mix[0] for item in pbg_mix]
ratio_arr_sage = [item/sage[0] for item in sage]
ratio_arr_best_fit = [item/best_fit[0] for item in best_fit]

if zhongzuobiao_type == "job_num":
    iterative_his_gamma_0_0 = np.array([460, 646, 641, 639])
    his_gamma_0_0 = np.array([308, 286, 300, 390])
    pbg = np.array([381, 292, 350, 380])
    pbg_mix = np.array([434, 384, 460, 500])
    sage = np.array([383, 298, 350, 380])
    best_fit = np.array([376, 287, 350, 380])
    
elif zhongzuobiao_type == "significance_all":
    iterative_his_gamma_0_0 = np.array([3.519885838242828, 2.3935149486104024, 0.0, 0.0])
    iterative_his_gamma_0_0[2] = ratio_arr_iterative_his_gamma_0_0[2] * iterative_his_gamma_0_0[0]
    iterative_his_gamma_0_0[3] = ratio_arr_iterative_his_gamma_0_0[3] * iterative_his_gamma_0_0[0]
    his_gamma_0_0 = np.array([2.1349308881915987, 0.8676523357453894, 0.0, 0.0])
    his_gamma_0_0[2] = ratio_arr_his_gamma_0_0[2] * his_gamma_0_0[0]
    his_gamma_0_0[3] = ratio_arr_his_gamma_0_0[3] * his_gamma_0_0[0]
    pbg = np.array([2.5968883436859636, 1.0067507564357072, 0.0, 0.0])
    pbg[2] = ratio_arr_pbg[2] * pbg[0]
    pbg[3] = ratio_arr_pbg[3] * pbg[0]
    pbg_mix = np.array([2.6892566198130097, 1.0509979228035369, 0.0, 0.0])
    pbg_mix[2] = ratio_arr_pbg_mix[2] * pbg_mix[0]
    pbg_mix[3] = ratio_arr_pbg_mix[3] * pbg_mix[0]
    sage = np.array([2.60727342149078, 1.013429247246415, 0.0, 0.0])
    sage[2] = ratio_arr_sage[2] * sage[0]
    sage[3] = ratio_arr_sage[3] * sage[0]
    best_fit = np.array([2.68622938012295, 1.0255629202420589, 0.0, 0.0])
    best_fit[2] = ratio_arr_best_fit[2] * best_fit[0]
    best_fit[3] = ratio_arr_best_fit[3] * best_fit[0]
    
elif zhongzuobiao_type == "significance_success":
    iterative_his_gamma_0_0 = np.array([7.6519257353104955, 7.410262998793815, 0.0, 0.0])
    iterative_his_gamma_0_0[2] = ratio_arr_iterative_his_gamma_0_0[2] * iterative_his_gamma_0_0[0]
    iterative_his_gamma_0_0[3] = ratio_arr_iterative_his_gamma_0_0[3] * iterative_his_gamma_0_0[0]
    his_gamma_0_0 = np.array([6.931593792829865, 6.067498851366359, 0.0, 0.0])
    his_gamma_0_0[2] = ratio_arr_his_gamma_0_0[2] * his_gamma_0_0[0]
    his_gamma_0_0[3] = ratio_arr_his_gamma_0_0[3] * his_gamma_0_0[0]
    pbg = np.array([6.815979904687569, 6.895553126271968, 0.0, 0.0])
    pbg[2] = ratio_arr_pbg[2] * pbg[0]
    pbg[3] = ratio_arr_pbg[3] * pbg[0]
    pbg_mix = np.array([6.196443824453939, 5.473947514601754, 0.0, 0.0])
    pbg_mix[2] = ratio_arr_pbg_mix[2] * pbg_mix[0]
    pbg_mix[3] = ratio_arr_pbg_mix[3] * pbg_mix[0]
    sage = np.array([6.807502405981148, 6.801538572123591, 0.0, 0.0])
    sage[2] = ratio_arr_sage[2] * sage[0]
    sage[3] = ratio_arr_sage[3] * sage[0]
    best_fit = np.array([7.144227074795079, 7.146779932000411, 0.0, 0.0])
    best_fit[2] = ratio_arr_best_fit[2] * best_fit[0]
    best_fit[3] = ratio_arr_best_fit[3] * best_fit[0]

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
fig = plt.figure(figsize=(10, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框


plt.plot(henzuobiaos, iterative_his_gamma_0_0, marker='o', color="r", label=r"IterativeHIS:$\gamma$-0.0", linewidth=1.5)
plt.plot(henzuobiaos, his_gamma_0_0, marker='o', color="g", label=r"HIS:$\gamma$-0.0", linewidth=1.5)
plt.plot(henzuobiaos, pbg, marker='o', color="c", label=r"PBG", linewidth=1.5)
plt.plot(henzuobiaos, pbg_mix, marker='o', color="b", label=r"PBG_MIX", linewidth=1.5)
plt.plot(henzuobiaos, sage, marker='o', color="xkcd:violet", label=r"SAGE", linewidth=1.5)
plt.plot(henzuobiaos, best_fit, marker='o', color="xkcd:orange", label=r"BestFit", linewidth=1.5)


group_labels = list(str(hen) for hen in henzuobiaos)  # x轴刻度的标识
plt.xticks(henzuobiaos, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel(r"Number of test jobs $n$", fontsize=13, fontweight='bold')
if zhongzuobiao_type == "job_num":
    plt.ylabel("Number of allocated jobs", fontsize=13, fontweight='bold')
elif zhongzuobiao_type == "significance_all":
    plt.ylabel("Average significance of all jobs", fontsize=13, fontweight='bold')
elif zhongzuobiao_type == "significance_success":
    plt.ylabel("Average significance of allocated jobs", fontsize=13, fontweight='bold')
# plt.xlim(0.9, 6.1)  # 设置x轴的范围
# plt.ylim(1.5, 16)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
plt.legend(loc=2, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
plt.subplots_adjust(left=0.1, right=0.88)

path = './fig_change_online_job_num_{}'.format(zhongzuobiao_type)
plt.savefig(path + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(path + '.pdf')
pp.savefig(fig)
pp.close()