import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_style("ticks")

data = np.load("Fig2_PT1_varyhab.npz")


# Plotting the player categories---------------------------
ax = plt.axes()
ax.plot(data["h"], data["SCmean"], 'bo-', label = 'SC')
ax.fill_between(data["h"], data["SCmean"]-data["SCstd"],
                 data["SCmean"]+data["SCstd"], alpha=0.3, color="b")

ax.plot(data["h"], data["UCmean"], 'o-', label = 'UC', color="g")
ax.fill_between(data["h"], data["UCmean"]-data["UCstd"],
                 data["UCmean"]+data["UCstd"], alpha=0.3, color="g")

ax.plot(data["h"], data["UDmean"], 'o-', label = 'UD', color="peru")
ax.fill_between(data["h"], data["UDmean"]-data["UDstd"],
                 data["UDmean"]+data["UDstd"], alpha=0.3, color="peru")

ax.plot(data["h"], data["SDmean"], 'o-', label = 'SD', color="r")
ax.fill_between(data["h"], data["SDmean"]-data["SDstd"],
                 data["SDmean"]+data["SDstd"], alpha=0.3, color="r")

ax.set_xlabel('habituation'+r'($h$)', fontsize=15)
ax.set_ylabel('fraction of players in each category', fontsize=15)
ax.legend(loc=(0, 0.5), framealpha=0, fontsize=15)
ax.set_yticks([0.05, 0.15, 0.25, 0.35, 0.45])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
#ax.grid(None)
plt.subplots_adjust(right=.98, left=0.15, bottom=0.13, top=0.96)
plt.savefig('Fig2_PT1_satactVShab_beta'+str(1)+'.tif', dpi=300)
plt.show()
#--------------------------------------------------------------


# FIXED STATE TO FOUR STATES ---------------
# Domain 1 
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

plt.plot(data["h"], data["UCtSCmean"], 'o-',
         label = r'UC$\Rightarrow$SC', color='b')
plt.fill_between(data["h"], data["UCtSCmean"]-data["UCtSCstd"],
                 data["UCtSCmean"]+data["UCtSCstd"],
                 alpha=0.3, color="b")

plt.plot(data["h"], data["UDtUCmean"], 'o-',
         label = r'UD$\Rightarrow$UC', color="g")
plt.fill_between(data["h"], data["UDtUCmean"]-data["UDtUCstd"],
                data["UDtUCmean"]+data["UDtUCstd"],
                 alpha=0.3, color="g")

plt.subplots_adjust(right=.98, left=0.15, bottom=0.13, top=0.96)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('dominant category transitions', fontsize=15)
plt.legend(framealpha=0, fontsize=15)
plt.ylim(-10,140)
#plt.grid(None)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('Fig2_PT1_Domain1TransitionsVShab_beta'+str(1)+'.tif', dpi=300)
plt.show()


#Domain 2
plt.plot(data["h"], data["SDtSDmean"], 'o-',
         label = r'SD$\Rightarrow$SD', color='k')
plt.fill_between(data["h"], data["SDtSDmean"]-data["SDtSDstd"],
                data["SDtSDmean"]+data["SDtSDstd"],
                 alpha=0.3, color="k")

plt.plot(data["h"], data["SDtUCmean"], 'o-',
         label = r'SD$\Rightarrow$UC', color="g")
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

plt.subplots_adjust(right=.98, left=0.15, bottom=0.13, top=0.96)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('dominant category transitions', fontsize=15)
plt.legend(loc=(0.68,0.0), framealpha=0, fontsize=15)
plt.ylim(-10,140)
#plt.grid(None)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('Fig2_PT1_Domain2TransitionsVShab_beta'+str(1)+'.tif', dpi=300)
plt.show()
