import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_style("ticks")


data = np.load("PT1_arrays_hab_beta1.0_re1.0.npz")
dataraw = np.load("rawdata_PT1_arrays_hab_beta1.0_re1.0.npz")

eqbTime=500
totNP=500

SC_arr = dataraw['SC']
SD_arr = dataraw['SD']
UC_arr = dataraw['UC']
UD_arr = dataraw['UD']

SC_trials = np.mean(SC_arr[:,:,eqbTime:], axis=-1)
UC_trials = np.mean(UC_arr[:,:,eqbTime:], axis=-1)
SD_trials = np.mean(SD_arr[:,:,eqbTime:], axis=-1)
UD_trials = np.mean(UD_arr[:,:,eqbTime:], axis=-1)
SCmean = np.mean(SC_trials, axis=-1)/totNP
UCmean = np.mean(UC_trials, axis=-1)/totNP
SDmean = np.mean(SD_trials, axis=-1)/totNP
UDmean = np.mean(UD_trials, axis=-1)/totNP
SCstd = np.std(SC_trials, axis=-1)/totNP
UCstd = np.std(UC_trials, axis=-1)/totNP
SDstd = np.std(SD_trials, axis=-1)/totNP
UDstd = np.std(UD_trials, axis=-1)/totNP

# FIXED STATE TO FOUR STATES ---------------
# Cyc1
plt.clf()
plt.plot(data["h"], data["SDtSDmean"], 'o-',
         label = r'SD$\Rightarrow$SD', color='k')
plt.fill_between(data["h"], data["SDtSDmean"]-data["SDtSDstd"],
                 data["SDtSDmean"]+data["SDtSDstd"],
                 alpha=0.3, color="k")
plt.plot(data["h"], data["SDtUCmean"], 'o-', label =
         r'SD$\Rightarrow$UC', color="g")
plt.fill_between(data["h"], data["SDtUCmean"]-data["SDtUCstd"],
                 data["SDtUCmean"]+data["SDtUCstd"],
                 alpha=0.3, color="g")
plt.plot(data["h"], data["UCtUDmean"], 'o-',
         label = r'UC$\Rightarrow$UD', color='peru')
plt.fill_between(data["h"], data["UCtUDmean"]-data["UCtUDstd"],
                 data["UCtUDmean"]+data["UCtUDstd"],
                 alpha=0.3, color="peru")
plt.plot(data["h"], data["UDtSDmean"], 'o-',
         label = r'UD$\Rightarrow$SD', color='r')
plt.fill_between(data["h"], data["UDtSDmean"]-data["UDtSDstd"],
                data["UDtSDmean"]+data["UDtSDstd"],
                 alpha=0.3, color="r")

plt.subplots_adjust(right=.98, left=0.15, bottom=0.13, top=0.98)
#plt.grid(None)
plt.xlabel('habituation($h$)', fontsize=15)
plt.ylabel('average number of transitions per round', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc=(0.05,0.65), framealpha=0, fontsize=15)
plt.savefig('Fig7_cyc2_beta'+
            str(1)+'_re'+str(1.0)+'.tif', dpi=300)
plt.show()


plt.clf()
plt.plot(data["h"], data["SCtSCmean"], 'o-',
         label = r'SC$\Rightarrow$SC', color='k')
plt.fill_between(data["h"], data["SCtSCmean"]-data["SCtSCstd"],
                 data["SCtSCmean"]+data["SCtSCstd"],
                 alpha=0.3, color="k")
plt.plot(data["h"], data["SCtUDmean"], 'o-',
         label = r'SC$\Rightarrow$UD', color='peru')
plt.fill_between(data["h"], data["SCtUDmean"]-data["SCtUDstd"],
                 data["SCtUDmean"]+data["SCtUDstd"],
                 alpha=0.3, color="peru")
plt.plot(data["h"], data["UDtUCmean"], 'o-',
         label = r'UD$\Rightarrow$UC', color="g")
plt.fill_between(data["h"], data["UDtUCmean"]-data["UDtUCstd"],
                data["UDtUCmean"]+data["UDtUCstd"],
                 alpha=0.3, color="g")
plt.plot(data["h"], data["UCtSCmean"], 'o-',
         label = r'UC$\Rightarrow$SC', color='b')
plt.fill_between(data["h"], data["UCtSCmean"]-data["UCtSCstd"],
                 data["UCtSCmean"]+data["UCtSCstd"],
                 alpha=0.3, color="b")

plt.subplots_adjust(right=.98, left=0.15, bottom=0.13, top=0.98)
#plt.grid(None)
plt.xlabel('habituation($h$)', fontsize=15)
plt.ylabel('average number of transitions per round', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc=(0.0,0.65), framealpha=0, fontsize=15)
plt.savefig('Fig7_cyc1_beta'+
            str(1)+'_re'+str(1.0)+'.tif', dpi=300)
plt.show()

