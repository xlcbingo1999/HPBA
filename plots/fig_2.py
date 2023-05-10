# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

henzuobiaos = np.array([0.1, 0.4, 0.7, 1.0])

zhongzuobiao_type = "significance_success" # "job_num", "significance_all" "significance_success"

# iterative_his_gamma_0_0 = np.array([459.4, 472.4, 641, 639])
# his_gamma_0_0 = np.array([308, 286, 300, 390])
# pbg = np.array([381, 292, 350, 380])
# pbg_mix = np.array([434, 384, 460, 500])
# sage = np.array([383, 298, 350, 380])
# best_fit = np.array([376, 287, 350, 380])

if zhongzuobiao_type == "job_num":
    iterative_his_gamma_0_0 = np.array([459.4, 301, 296, 285])
    # iterative_his_gamma_0_0_min = np.array([453, 646, 641, 639])
    # iterative_his_gamma_0_0_max = np.array([471, 646, 641, 639])

    # his_gamma_0_0 = np.array([407, 286, 300, 390])
    # his_gamma_0_0_min = np.array([308, 286, 300, 390])
    # his_gamma_0_0_max = np.array([308, 286, 300, 390])

    pbg = np.array([379, 158, 135, 180])
    # pbg_min = np.array([381, 292, 350, 380])
    # pbg_max = np.array([381, 292, 350, 380])

    pbg_mix = np.array([423, 220, 211, 253])
    # pbg_mix_min = np.array([434, 384, 460, 500])
    # pbg_mix_max = np.array([434, 384, 460, 500])

    sage = np.array([379, 157, 126, 180])
    # sage_min = np.array([383, 298, 350, 380])
    # sage_max = np.array([383, 298, 350, 380])

    best_fit = np.array([376, 182, 156, 227])
    # best_fit_min = np.array([376, 287, 350, 380])
    # best_fit_max = np.array([376, 287, 350, 380])
    
elif zhongzuobiao_type == "significance_all":
    iterative_his_gamma_0_0 = np.array([3.535938795146002, 1.668098056480535, 1.5392849081070703, 1.734900686795597])
    # iterative_his_gamma_0_0_min = np.array([460, 646, 641, 639])
    # iterative_his_gamma_0_0_max = np.array([460, 646, 641, 639])

    # his_gamma_0_0 = np.array([2.5810112784964114, 286, 300, 390])
    # his_gamma_0_0_min = np.array([308, 286, 300, 390])
    # his_gamma_0_0_max = np.array([308, 286, 300, 390])

    pbg = np.array([2.5805945344518455, 1.078856661436988, 0.7753441902535858, 1.1795184746413931])
    # pbg_min = np.array([381, 292, 350, 380])
    # pbg_max = np.array([381, 292, 350, 380])

    pbg_mix = np.array([2.6477434042008205, 1.1530471391521513, 0.9277808737832992, 1.2845103019339146])
    # pbg_mix_min = np.array([434, 384, 460, 500])
    # pbg_mix_max = np.array([434, 384, 460, 500])

    sage = np.array([2.6028358334400648, 1.0703841332607578, 0.745426685771004, 1.172191774462089])
    # sage_min = np.array([383, 298, 350, 380])
    # sage_max = np.array([383, 298, 350, 380])

    best_fit = np.array([2.68622938012295, 1.2593551005379096, 0.8903225537909832, 1.5123030705686467])
    # best_fit_min = np.array([376, 287, 350, 380])
    # best_fit_max = np.array([376, 287, 350, 380])
    
elif zhongzuobiao_type == "significance_success":
    iterative_his_gamma_0_0 = np.array([7.697777133886605, 5.54185400824098, 5.200286851713075, 6.087370830861744])
    # iterative_his_gamma_0_0_min = np.array([460, 646, 641, 639])
    # iterative_his_gamma_0_0_max = np.array([460, 646, 641, 639])

    # his_gamma_0_0 = np.array([6.341551052816735, 286, 300, 390])
    # his_gamma_0_0_min = np.array([308, 286, 300, 390])
    # his_gamma_0_0_max = np.array([308, 286, 300, 390])

    pbg = np.array([6.808956555281914, 6.828206717955621, 5.74329029817471, 5.671118371078297])
    # pbg_min = np.array([381, 292, 350, 380])
    # pbg_max = np.array([381, 292, 350, 380])

    pbg_mix = np.array([6.259440671869552, 5.241123359782505, 6.696701226027571, 5.077115817920611])
    # pbg_mix_min = np.array([434, 384, 460, 500])
    # pbg_mix_max = np.array([434, 384, 460, 500])

    sage = np.array([6.867640721477743, 6.817733332871069, 5.9160848077063815, 6.512176524789384])
    # sage_min = np.array([383, 298, 350, 380])
    # sage_max = np.array([383, 298, 350, 380])

    best_fit = np.array([7.144227074795079, 6.919533519439063, 5.707195857634508, 6.66212806417906])
    # best_fit_min = np.array([376, 287, 350, 380])
    # best_fit_max = np.array([376, 287, 350, 380])

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
# if zhongzuobiao_type == "job_num":
#     plt.fill_between(henzuobiaos, iterative_his_gamma_0_0_min, iterative_his_gamma_0_0_max, color="r", alpha=0.5)


# plt.plot(henzuobiaos, his_gamma_0_0, marker='o', color="g", label=r"HIS:$\gamma$-0.0", linewidth=1.5)
# plt.fill_between(henzuobiaos, his_gamma_0_0_min, his_gamma_0_0_max, color="g", alpha=0.5)

plt.plot(henzuobiaos, pbg, marker='o', color="c", label=r"PBG", linewidth=1.5)
# plt.fill_between(henzuobiaos, pbg_min, pbg_max, color="c", alpha=0.5)

plt.plot(henzuobiaos, pbg_mix, marker='o', color="b", label=r"PBG_MIX", linewidth=1.5)
# plt.fill_between(henzuobiaos, pbg_mix_min, pbg_mix_max, color="b", alpha=0.5)

plt.plot(henzuobiaos, sage, marker='o', color="xkcd:violet", label=r"SAGE", linewidth=1.5)
# plt.fill_between(henzuobiaos, sage_min, sage_max, color="xkcd:violet", alpha=0.5)

plt.plot(henzuobiaos, best_fit, marker='o', color="xkcd:orange", label=r"BestFit", linewidth=1.5)
# plt.fill_between(henzuobiaos, best_fit_min, best_fit_max, color="xkcd:orange", alpha=0.5)



group_labels = list(str(hen) for hen in henzuobiaos)  # x轴刻度的标识
plt.xticks(henzuobiaos, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel(r"Require-supply Ratio $\lambda$", fontsize=13, fontweight='bold')
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

path = '/home/netlab/DL_lab/opacus_testbed/plots/fig2{}'.format(zhongzuobiao_type)
plt.savefig(path + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(path + '.pdf')
pp.savefig(fig)
pp.close()