import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rc('axes', labelsize=8)
mpl.rc('ytick', labelsize=8)
sns.set(style="white")
sns.set_style("ticks")

data = np.load("Arrays_hab0.4_re0.3.npz")
trounds = data["rounds"]
eqbTime = 700
rounds = 750
hab = 0.4
Re=0.3
'''
SCpay_arr = data["SCpay"]
SDpay_arr = data["SDpay"]
UCpay_arr = data["UCpay"]
UDpay_arr = data["UDpay"]

SCasp_arr = data["SCasp"]
SDasp_arr = data["SDasp"]
UCasp_arr = data["UCasp"]
UDasp_arr = data["UDasp"]
'''

CC_arr = data["CC"]
CD_arr = data["CD"]
DD_arr = data["DD"]

Cdeg_arr = data["Cdeg"]
Ddeg_arr = data["Ddeg"]

coopfrac_arr = data["cfrac"]

ax2 = plt.axes()
ax1 = ax2.twinx()
ax1.plot(trounds[eqbTime:],
         coopfrac_arr[0,eqbTime:],
         'co-', linewidth=1, label='C-frac')
ax2.plot(trounds[eqbTime:], Cdeg_arr[0,eqbTime:], 'bo-',
         label='<deg(C)>')
ax2.plot(trounds[eqbTime:], Ddeg_arr[0,eqbTime:], 'ro-',
         label='<deg(D)>')
ax1.legend(loc=(0.82,1.0), fontsize=15, framealpha=0)
ax2.legend(loc=(-0.1,1.0), fontsize=15, framealpha=0, ncol=2)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax1.grid(False)
ax2.grid(False)
ax2.set_xlabel("round", fontsize=15)
#plt.xticks(range(eqbTime, rounds-1, 3))
plt.subplots_adjust(right=.9, left=0.15, bottom=0.13)
#plt.savefig("Deg_reNet_re"+str(Re)+"_h"+str(hab)+".tif", dpi=300)
plt.savefig("Deg_reNet_re"+str(Re)+"_h"+str(hab)+".jpg", dpi=300)
plt.show()

plt.clf()
ax2 = plt.axes()
ax1 = ax2.twinx()
ax1.plot(trounds[eqbTime:],
         coopfrac_arr[0,eqbTime:],
         'co-', linewidth=1, label='C-frac')
plt.xlabel('rounds')
ax2.plot(trounds[eqbTime:], CC_arr[0,eqbTime:], 'bo-',
         label='C-C')
ax2.plot(trounds[eqbTime:], DD_arr[0,eqbTime:], 'ro-',
         label='D-D')
ax2.plot(trounds[eqbTime:], CD_arr[0,eqbTime:], 'ko-',
         label='C-D')
ax1.legend(loc=(0.82,1.0), fontsize=15, framealpha=0)
ax2.legend(loc=(-0.1,1.0), fontsize=15, framealpha=0, ncol=3)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax1.grid(False)
ax2.grid(False)
ax2.set_xlabel("round", fontsize=15)
plt.subplots_adjust(right=0.9, left=0.15, bottom=0.13, top=0.9)
#plt.savefig("Links_reNet_re"+str(Re)+"_h"+str(hab)+".tif")
plt.savefig("Links_reNet_re"+str(Re)+"_h"+str(hab)+".jpg")
plt.show()
