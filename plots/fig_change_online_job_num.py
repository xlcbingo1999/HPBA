# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

henzuobiaos = np.array([1000, 2000, 3000, 4000])


zhongzuobiao_type = "significance_all" # "job_num", "significance_all" "significance_success"

if zhongzuobiao_type == "job_num":
    iterative_his_gamma_0_0 = np.array([459.4, 646, 641, 639])
    iterative_his_gamma_0_0_min = np.array([453, 646, 641, 639])
    iterative_his_gamma_0_0_max = np.array([471, 646, 641, 639])

    his_gamma_0_0 = np.array([407, 286, 300, 390])
    his_gamma_0_0_min = np.array([308, 286, 300, 390])
    his_gamma_0_0_max = np.array([308, 286, 300, 390])

    pbg = np.array([379, 292, 329, 435])
    pbg_min = np.array([381, 292, 350, 380])
    pbg_max = np.array([381, 292, 350, 380])

    pbg_mix = np.array([423, 384, 379, 505])
    pbg_mix_min = np.array([434, 384, 460, 500])
    pbg_mix_max = np.array([434, 384, 460, 500])

    sage = np.array([379, 298, 337, 437])
    sage_min = np.array([383, 298, 350, 380])
    sage_max = np.array([383, 298, 350, 380])

    best_fit = np.array([376, 287, 323, 422])
    best_fit_min = np.array([376, 287, 350, 380])
    best_fit_max = np.array([376, 287, 350, 380])
    
elif zhongzuobiao_type == "significance_all":
    iterative_his_gamma_0_0 = np.array([3.535938795146002, 2.3935149486104024, 1.9638429415343266, 1.4755733102266904])
    iterative_his_gamma_0_0_min = np.array([460, 646, 641, 639])
    iterative_his_gamma_0_0_max = np.array([460, 646, 641, 639])

    his_gamma_0_0 = np.array([2.5810112784964114, 286, 300, 390])
    his_gamma_0_0_min = np.array([308, 286, 300, 390])
    his_gamma_0_0_max = np.array([308, 286, 300, 390])

    pbg = np.array([2.5805945344518455, 1.0067507564357072, 0.8057246627390711, 0.6167341228547648])
    pbg_min = np.array([381, 292, 350, 380])
    pbg_max = np.array([381, 292, 350, 380])

    pbg_mix = np.array([2.6477434042008205, 1.0509979228035369, 0.8460165882214832, 0.6327319676133453])
    pbg_mix_min = np.array([434, 384, 460, 500])
    pbg_mix_max = np.array([434, 384, 460, 500])

    sage = np.array([2.6028358334400648, 1.013429247246415, 0.8101928203978823, 0.6161615150326591])
    sage_min = np.array([383, 298, 350, 380])
    sage_max = np.array([383, 298, 350, 380])

    best_fit = np.array([2.68622938012295, 1.0255629202420589, 0.8316688519253763, 0.6294322809938524])
    best_fit_min = np.array([376, 287, 350, 380])
    best_fit_max = np.array([376, 287, 350, 380])
    
elif zhongzuobiao_type == "significance_success":
    iterative_his_gamma_0_0 = np.array([7.697777133886605, 7.410262998793815, 9.191152612485148, 9.236765635221849])
    iterative_his_gamma_0_0_min = np.array([460, 646, 641, 639])
    iterative_his_gamma_0_0_max = np.array([460, 646, 641, 639])

    his_gamma_0_0 = np.array([6.341551052816735, 286, 300, 390])
    his_gamma_0_0_min = np.array([308, 286, 300, 390])
    his_gamma_0_0_max = np.array([308, 286, 300, 390])

    pbg = np.array([6.808956555281914, 6.895553126271968, 7.347033398836515, 5.671118371078297])
    pbg_min = np.array([381, 292, 350, 380])
    pbg_max = np.array([381, 292, 350, 380])

    pbg_mix = np.array([6.259440671869552, 5.473947514601754, 6.696701226027571, 5.011738357333428])
    pbg_mix_min = np.array([434, 384, 460, 500])
    pbg_mix_max = np.array([434, 384, 460, 500])

    sage = np.array([6.867640721477743, 6.801538572123591, 7.212398994639902, 5.63992233439505])
    sage_min = np.array([383, 298, 350, 380])
    sage_max = np.array([383, 298, 350, 380])

    best_fit = np.array([7.144227074795079, 7.146779932000411, 7.724478500854889, 5.9661827582355675])
    best_fit_min = np.array([376, 287, 350, 380])
    best_fit_max = np.array([376, 287, 350, 380])

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
plt.fill_between(henzuobiaos, iterative_his_gamma_0_0_min, iterative_his_gamma_0_0_max, color="r", alpha=0.5)

plt.plot(henzuobiaos, his_gamma_0_0, marker='o', color="g", label=r"HIS:$\gamma$-0.0", linewidth=1.5)
plt.fill_between(henzuobiaos, his_gamma_0_0_min, his_gamma_0_0_max, color="g", alpha=0.5)

plt.plot(henzuobiaos, pbg, marker='o', color="c", label=r"PBG", linewidth=1.5)
plt.fill_between(henzuobiaos, pbg_min, pbg_max, color="c", alpha=0.5)

plt.plot(henzuobiaos, pbg_mix, marker='o', color="b", label=r"PBG_MIX", linewidth=1.5)
plt.fill_between(henzuobiaos, pbg_mix_min, pbg_mix_max, color="b", alpha=0.5)

plt.plot(henzuobiaos, sage, marker='o', color="xkcd:violet", label=r"SAGE", linewidth=1.5)
plt.fill_between(henzuobiaos, sage_min, sage_max, color="xkcd:violet", alpha=0.5)

plt.plot(henzuobiaos, best_fit, marker='o', color="xkcd:orange", label=r"BestFit", linewidth=1.5)
plt.fill_between(henzuobiaos, best_fit_min, best_fit_max, color="xkcd:orange", alpha=0.5)

plt.plot(henzuobiaos, best_fit, marker='o', color="xkcd:", label=r"BestFit", linewidth=1.5)
plt.fill_between(henzuobiaos, best_fit_min, best_fit_max, color="xkcd:orange", alpha=0.5)


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

path = '/home/netlab/DL_lab/opacus_testbed/plots/fig_change_online_job_num_{}'.format(zhongzuobiao_type)
plt.savefig(path + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(path + '.pdf')
pp.savefig(fig)
pp.close()