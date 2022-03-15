import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib as mpl
mpl.rc('axes', labelsize=8)
mpl.rc('ytick', labelsize=8)
sns.set(style="white")
sns.set_style("ticks")
#matplotlib.use("Agg")

data = np.load("Arrays_hab0.4_re0.3.npz")
deg_distribution = data["deg_dist"]

hab=0.4
Re=0.3

Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
rounds=750
'''
fig, ax = plt.subplots(figsize=(8,5))
def update(frame):
    plt.clf()
    plt.xlim(0,500)
    plt.hist(deg_distribution[frame,:], bins=25)
    plt.title("round="+str(frame)+
              #" C-frac="+str(coopfrac_arr[0,frame])+
              " h="+str(hab)+
              " re="+str(Re))
Fr = np.arange(rounds-100, rounds, 1)
ani=matplotlib.animation.FuncAnimation(fig, update, frames=Fr,
                                       interval=1000, repeat=False)
ani.save("P(deg)_HI_h"+str(hab)+"_re"+str(Re)+".mp4",
         writer=writer, dpi=300)

'''
frame1=720
frame2=717
fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
ax1.hist(deg_distribution[frame1,:], bins=50, density=True,
         histtype="step", label="round 720", color='b')
ax2.hist(deg_distribution[frame2,:], bins=50, density=True,
         histtype="step", label="round 717", color='r')
ax2.legend(framealpha=0)
ax1.legend(framealpha=0)
#linestyle='dashed',color=('k'))
'''
plt.title("round="+str(frame)+
          #" C-frac="+str(coopfrac_arr[0,frame])+
          " h="+str(hab)+
          " re="+str(Re))
'''
ax2.set_ylabel(r'$P(k)$'+' (degree distribution)', fontsize=15)
ax2.yaxis.set_label_coords(-0.1, 1)
ax2.set_xlabel(r'$k$'+' (degree)', fontsize=15)
plt.subplots_adjust(right=.96, left=0.15, bottom=0.13, top=0.95)
ax2.set_xlim((0,500))
ax1.set_xlim((0,500))
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
plt.savefig("S2-degreedistribution.png")
plt.savefig("S2-degreedistribution.jpg")
plt.show()
