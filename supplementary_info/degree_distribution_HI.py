import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.sparse as scsp
import networkx as nx
import time
from datetime import datetime
import numba as nb
from numba import int64, float64
import seaborn as sns
import PGG
import Record_vectorized as Record
import rewire
import initialize as init
import features
import eval_shift
#matplotlib.use("Agg")
#import windowing

sns.set(style="whitegrid")
sns.set_style("ticks")

@nb.njit
def eval_LinksDeg(AM, res, N):
    CC, CD, DD = 0, 0, 0
    C_deg, D_deg = 0, 0
    
    pl = -1 #player index
    C_count = 0
    for FPAct in res:
        pl = pl + 1
        Neigh = np.nonzero(AM[pl])[0]
        C_deg = C_deg + len(Neigh)*FPAct[0]
        D_deg = D_deg + len(Neigh)*(1-FPAct[0])
        CNeigh = len(np.where(res[Neigh]==1)[0])
        DNeigh = len(np.where(res[Neigh]==0)[0])
        CC = CC + CNeigh*FPAct[0]
        DD = DD + DNeigh*(1-FPAct[0])
        CD = CD + CNeigh*(1-FPAct[0]) + DNeigh*(FPAct[0])
        C_count = C_count + FPAct[0]
    C_deg = C_deg/(C_count + 0.00001)
    D_deg = D_deg/(N - C_count + 0.00001)
    return(CC/2., CD/2., DD/2., C_deg, D_deg)


tTime = time.time()
totTime = 0.

totNP = 500
trials = 1
rounds = 750
trounds = np.arange(0,rounds,1)

eps = 0.02
c = 1.
r = 2.
b = r*c
ini_re = 0.3
beta_arr = np.array([10**(0.)])/c
hab = 0.9
eqbTime = 500
var_arr = beta_arr
Re = 0.3
print(len(beta_arr))

coopfrac_arr = np.zeros((trials, rounds))
sat_arr = np.zeros((trials, rounds))
deg_arr = np.zeros((trials, rounds))
Cdeg_arr = np.zeros((trials, rounds))
Ddeg_arr = np.zeros((trials, rounds))
CC_arr = np.zeros((trials, rounds))
DD_arr = np.zeros((trials, rounds))
CD_arr = np.zeros((trials, rounds))
SC_arr = np.zeros((trials, rounds))
UC_arr = np.zeros((trials, rounds))
SD_arr = np.zeros((trials, rounds))
UD_arr = np.zeros((trials, rounds))
satSC_arr = np.zeros((trials, rounds))
category_arr = np.zeros((len(var_arr), trials, rounds, totNP, 1))
deg_distribution = np.zeros((rounds, totNP))
clustering_arr = np.zeros((trials, rounds))

varInd = -1
for bet in beta_arr:
    varInd = varInd + 1
    hTime = time.time()
    print('beta=', bet)
    for it in range(trials):
        #print('trial number : '+str(it))
        AdjMat = init.init_adjmat(totNP, ini_re)
        [aspA, satA,
         pcA, payA,
         cpayA, habA,
         betaA, actA] = init.init_arr(totNP, AdjMat, 
                                      bet, hab, r, c)

        for i_main in range(rounds):

            print(i_main)
            pay = PGG.game(AdjMat, actA, r, c, totNP)

            G = nx.convert_matrix.from_numpy_array(AdjMat)
            clustering_dict = nx.algorithms.cluster.clustering(G)
            clustering_arr[it, i_main
                           ] = sum(clustering_dict.values())/totNP
            print(sum(clustering_dict.values())/totNP)
            [C_C, C_D, D_D, Cdeg, Ddeg
             ] = eval_LinksDeg(AdjMat, actA, totNP)
            Cdeg_arr[it, i_main] = Cdeg
            Ddeg_arr[it, i_main] = Ddeg
            CC_arr[it, i_main] = C_C#/(C_C + D_D + C_D)
            DD_arr[it, i_main] = D_D#/(C_C + D_D + C_D)
            CD_arr[it, i_main] = C_D#/(C_C + D_D + C_D)
            
            coopfrac_arr[it,i_main] = np.sum(actA[:,0])/totNP
            deg_distribution[i_main,:] = np.sum(AdjMat, axis=1)
            AdjMat_P = AdjMat.copy()
            deg_arr[it, i_main] = np.sum(np.sum(AdjMat))/2.
            AdjMat = rewire.rewiring_process(AdjMat, actA, Re)
            
            # The asp  will be used to determine sat in next round
            [aspA, satA,
             pcA, actA,
             cpayA] = Record.update_iterated(pay, aspA,
                                             satA, payA,
                                             cpayA, pcA,
                                             habA, betaA,
                                             actA, eps,
                                             totNP)
            
    print('beta='+str(bet)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))
    
    np.savez("Arrays_hab"
             +str(hab)+"_re"+str(Re)+".npz",
             
             cfrac = coopfrac_arr, rounds = trounds,
             
             SC=SC_arr, SD=SD_arr,
             UC=UC_arr, UD=UD_arr,

             deg_dist = deg_distribution,

             clustering_coeff = clustering_arr,
             CC=CC_arr, CD=CD_arr, DD=DD_arr,
             Cdeg=Cdeg_arr, Ddeg=Ddeg_arr)

    '''
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    
    fig, ax = plt.subplots(figsize=(8,5))
    def update(frame):
        plt.clf()
        plt.xlim(0,500)
        plt.hist(deg_distribution[frame,:], bins=25)
        plt.title("round="+str(frame)+
                  " C-frac="+str(coopfrac_arr[0,frame])+
                  " h="+str(hab)+
                  " re="+str(Re))
    Fr = np.arange(rounds-100, rounds, 1)
    ani=matplotlib.animation.FuncAnimation(fig, update, frames=Fr,
                                           interval=1000, repeat=False)
    ani.save("P(deg)_HI_h"+str(hab)+"_re"+str(Re)+".mp4",
             writer=writer, dpi=300)


    #np.savez("clusteringcoeff_re"+str(Re)+"_h"+str(hab)+".npz",
    #h=hab, re=Re,
    #clustering = clustering_arr)
    ax2 = plt.axes()
    ax1 = ax2.twinx()
    ax2.plot(trounds[eqbTime:], clustering_arr[0,eqbTime:], "bo-",
             label="<clustering coeff>")
    ax1.plot(trounds[eqbTime:], Cdeg_arr[0,eqbTime:], "co-",
             label="<deg(C)>")
    plt.xlabel("round")
    ax1.legend(loc=(0.82,1.0), fontsize=15, framealpha=0)
    ax2.legend(loc=(0.0,1.0), fontsize=15, framealpha=0)
    plt.savefig("ClusteringCoeff_re"+str(Re)+"_h"+str(hab)+".png")
    plt.show()
    '''
