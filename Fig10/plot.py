import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



data = np.load("heatmap_resVShab.npz")
#data = np.load("revprobabs_heatmap_resVShab.npz")
probs = data["p"]
print(probs)
eqbTime = 250
c=1

re_arr = data["re"]
h_arr = data["h"]

#h_arr = h_arr[::-1]
xlabels = ['{:,.2f}'.format(x1) for x1 in h_arr[::1]]
ylabels = ['{:,.1f}'.format(y1) for y1 in re_arr[::1]]


# C-frac
coopfracheat = np.mean(np.mean(data["coopfrac"][:,:,:,eqbTime:],
                               axis=-1), axis=-1)
print(np.shape(data["coopfrac"]))
plt.clf()
#coopfracheat = coopfracheat[::-1]
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
#\n p11='+str(probs[0])
#+' p10='+str(probs[1])+' p01='+str(probs[2])
#+' p00='+str(probs[3]), fontsize=15)
plt.savefig('coopfracheat_rehab.tif', dpi=300)
plt.show()

Csatheat = np.mean(np.mean(data["Csat"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
Dsatheat = np.mean(np.mean(data["Dsat"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
satlim = max([np.abs(np.min(Csatheat)), np.abs(np.max(Csatheat)),
               np.abs(np.min(Dsatheat)), np.abs(np.max(Dsatheat))])

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
#\n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
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
#\n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
plt.savefig('Dsatheat_rehab.tif', dpi=300)
plt.show()


#CC links
CCheat = np.mean(np.mean(data["CC"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
plt.clf()
ax = sns.heatmap(CCheat,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='gray')
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
#plt.title('C-C links \n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
plt.savefig('CCheat_rehab.tif', dpi=300)
plt.show()



#CD links
CDheat = np.mean(np.mean(data["CD"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
plt.clf()
ax = sns.heatmap(CDheat,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='gray')
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
#plt.title('C-D links \n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
plt.savefig('CDheat_rehab.tif', dpi=300)
plt.show()



#DD links
DDheat = np.mean(np.mean(data["DD"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
plt.clf()
ax = sns.heatmap(DDheat,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='gray')
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
#plt.title('D-D Links \n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
plt.savefig('DDheat_rehab.tif', dpi=300)
plt.show()


#C deg
Cdegheat = np.mean(np.mean(data["Cdeg"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
plt.clf()
ax = sns.heatmap(Cdegheat,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='gray')
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
#plt.title('<deg(C)> \n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
plt.savefig('Cdegheat_rehab.tif', dpi=300)
plt.show()


#D deg
Ddegheat = np.mean(np.mean(data["Ddeg"][:,:,:,eqbTime:],
                           axis=-1), axis=-1)
plt.clf()
ax = sns.heatmap(Ddegheat,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='gray')
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
#plt.title('<deg(D)> \n p11='+str(probs[0])
          #+' p10='+str(probs[1])+' p01='+str(probs[2])
          #+' p00='+str(probs[3]), fontsize=15)
plt.savefig('Ddegheat_rehab.tif', dpi=300)
plt.show()
