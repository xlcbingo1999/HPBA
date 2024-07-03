import matplotlib.pyplot as plt
import numpy as np
# Mean value of QoE
x= range(6)
y=[-0.209,0.623,0.679,0.676,0.831,1.2371]
   
fig, ax = plt.subplots()

for i in range(6):
    ax.bar(x[i],y[i],width=0.5,hatch='/'*2,bottom=0.0)

reference_y = 0
ax.axhline(y=reference_y, color='black', linewidth=2.0, linestyle='--')  # 红色、粗细为2的横线

plt.xticks(range(6),('BB','RB', 'Pensieve', 'PensievesR', 'PatchsR', 'SpecialSR'))
plt.ylim()
plt.ylabel("QoE")

plt.savefig("/home/netlab/DL_lab/opacus_testbed/xlc_test.png")
# plt.show()