import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



data = np.load("heatmap_reVShab.npz")

eqbTime = 250
c=1

re_arr = np.arange(0, 1.001, 0.1)
re_arr = re_arr[::-1]
re_arr = np.round(re_arr, 3)

h_arr = np.arange(0.05, 1.0, 0.1)
h_arr = np.round(h_arr, 3)

coopfracheat = np.mean(np.mean(data["coopfrac"][:,:,:,eqbTime:],
                               axis=-1), axis=-1)
Csatheat = np.mean(np.mean(data["Csat"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
Dsatheat = np.mean(np.mean(data["Dsat"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
satlim = max([np.abs(np.min(Csatheat)), np.abs(np.max(Csatheat)),
               np.abs(np.min(Dsatheat)), np.abs(np.max(Dsatheat))])

#CCheat = np.mean(np.mean(data["CC"][:,:,:,eqbTime:],
#                         axis=-1), axis=-1)

xlabels = ['{:,.2f}'.format(x1) for x1 in h_arr[::1]]
ylabels = ['{:,.1f}'.format(y1) for y1 in re_arr[::1]]

# C-frac
plt.clf()
ax = sns.heatmap(coopfracheat, vmin=0, vmax=1, xticklabels=xlabels,
                 cmap='RdBu', yticklabels=ylabels)
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xticklabels(xlabels[::2], rotation='horizontal')
ax.set_yticklabels(ylabels[::2], rotation='horizontal')
plt.subplots_adjust(right=.98, left=0.13, bottom=0.13, top=0.93)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('rewiring fraction'+r'($re$)', fontsize=15)
#ax.yaxis.set_label_coords(-0.25, 0.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('cooperator fraction', fontsize=15)
plt.savefig('coopfracheat_rehab.tif', dpi=300)
plt.show()


# CSat
plt.clf()
ax = sns.heatmap(Csatheat, vmin=-1*satlim, vmax=satlim,
                 yticklabels=ylabels, xticklabels=xlabels, cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xticklabels(xlabels[::2], rotation='horizontal')
ax.set_yticklabels(ylabels[::2], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.98, left=0.13, bottom=0.13, top=0.93)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('average satisfaction of cooperators', fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('rewiring fraction'+r'($re$)', fontsize=15)
plt.savefig('Csatheat_rehab.tif', dpi=300)
plt.show()


#DSat
plt.clf()
ax = sns.heatmap(Dsatheat, vmin=-1*satlim, vmax=satlim,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xticklabels(xlabels[::2], rotation='horizontal')
ax.set_yticklabels(ylabels[::2], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.98, left=0.13, bottom=0.13, top=0.93)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('rewiring fraction'+r'($re$)', fontsize=15)
plt.title('average satisfaction of defectors', fontsize=15)
plt.savefig('Dsatheat_rehab.tif', dpi=300)
plt.show()

'''
#CC links
plt.clf()
ax = sns.heatmap(CCheat,
                 xticklabels=xlabels, yticklabels=ylabels,
                 cmap='binary')
ax.set_xticks(ax.get_xticks()[::2])
ax.set_yticks(ax.get_yticks()[::2])
ax.set_xticklabels(xlabels[::2], rotation='horizontal')
ax.set_yticklabels(ylabels[::2], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.98, left=0.13, bottom=0.13)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('rewiring fraction'+r'($re$)', fontsize=15)
plt.title('Number of C-C links', fontsize=15)
plt.savefig('CClinks_rehab.tif', dpi=300)
plt.show()
'''
