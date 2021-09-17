import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



data = np.load("Fig1abc_BetaHabHeatmap.npz")

eqbTime = 250
c=1

h_arr = data["h"]
beta_arr = data["beta"]

print(len(h_arr), len(beta_arr))
print(h_arr, beta_arr)
coopfracheat = np.mean(np.mean(data["coopfrac"][:,:,:,eqbTime:],
                               axis=-1), axis=-1)
Csatheat = np.mean(np.mean(data["Csat"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
Dsatheat = np.mean(np.mean(data["Dsat"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
satlim = max([np.abs(np.min(Csatheat)), np.abs(np.max(Csatheat)),
               np.abs(np.min(Dsatheat)), np.abs(np.max(Dsatheat))])

xlabels = ['{:,.1f}'.format(x1) for x1 in h_arr[::1]]
ylabels = ['{:,.1e}'.format(y1) for y1 in beta_arr[::1]]


# C-frac
plt.clf()
ax = sns.heatmap(coopfracheat, vmin=0, vmax=1, xticklabels=xlabels,
                 cmap='RdBu', yticklabels=ylabels)
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('sensitivity'+r'($\beta$)', fontsize=15)
ax.yaxis.set_label_coords(-0.25, 0.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('cooperator fraction', fontsize=15)
plt.subplots_adjust(right=.98, left=0.2, bottom=0.13, top=0.93)
plt.savefig('Fig1a_coopfracheat_betahab.tif', dpi=300)
plt.show()


# CSat
plt.clf()
ax = sns.heatmap(Csatheat, vmin=-1*satlim, vmax=satlim,
                 yticklabels=ylabels, xticklabels=xlabels, cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.98, left=0.2, bottom=0.13, top=0.93)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('average satisfaction of cooperators', fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('sensitivity'+r'($\beta$)', fontsize=15)
ax.yaxis.set_label_coords(-0.25, 0.5)
plt.savefig('Fig1b_Csatheat_betahab.tif', dpi=300)
plt.show()


#DSat
plt.clf()
ax = sns.heatmap(Dsatheat, vmin=-1*satlim, vmax=satlim,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.98, left=0.2, bottom=0.13, top=0.93)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('sensitivity'+r'($\beta$)', fontsize=15)
plt.title('average satisfaction of defectors', fontsize=15)
ax.yaxis.set_label_coords(-0.25, 0.5)
plt.savefig('Fig1c_Dsatheat_betahab.tif', dpi=300)
plt.show()
