import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as scsp
import networkx as nx
import pandas as pd
import time
from datetime import datetime
import numba as nb
from numba import int64, float64
import seaborn as sns
import PGG
import Record_vectorized as Record
import rewire
import initialize as init
import windowing
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter(useOffset=False)

#sns.set()


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
    C_deg = C_deg/(C_count + 0.0001)
    D_deg = D_deg/(N - C_count + 0.0001)
    return(CC/2., CD/2., DD/2., C_deg, D_deg)


tTime = time.time()
totTime = 0.

totNP = 500
trials = 2
rounds = 1000

eps = 0.02
c = 1.
r = 2.
b = r*c
ini_re = 0.3
bet = 1.

#re_arr = np.logspace(-5, -1, 5)
#re_arr = re_arr[::-1]
#re_arr = np.append(re_arr, 0)
re_arr = np.arange(0.0,1.01,0.1)
re_arr = re_arr[::-1]
re_arr = np.round(re_arr, 3)

h_arr = np.arange(0.05, 1.0, 0.1)
h_arr = np.round(h_arr, 3)

eqbTime = 500

print(h_arr)
print(re_arr)

coopfrac_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
sat_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
Csat_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
Dsat_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
degree_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
Cdeg_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
Ddeg_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
CC_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
DD_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))
CD_arr = np.zeros((len(re_arr), len(h_arr), trials, rounds))


hInd = -1
for hab in h_arr:

    hTime = time.time()
    print('h =', hab)
    hInd = hInd + 1

    reInd = -1
    for re in re_arr:

        reInd = reInd + 1
        print('re=', re)
        #print('hab=', hab)

        for it in range(trials):

            print("trials : ", it)
            
            AdjMat = init.init_adjmat(totNP, ini_re)
            [aspA, satA,
             pcA, payA,
             cpayA, habA,
             betaA, actA] = init.init_arr(totNP, AdjMat, 
                                          bet, hab, r, c)
            
            for i_main in range(rounds):
                 
                pay = PGG.game(AdjMat, actA, r, c, totNP)

                [coopfrac, sais, asp, Csat, Dsat,
                 Casp, Dasp] = Record.measure(actA, satA, aspA, AdjMat)
                coopfrac_arr[reInd, hInd, it, i_main] = coopfrac
                sat_arr[reInd, hInd, it, i_main] = sais
                Csat_arr[reInd, hInd, it, i_main] = Csat
                Dsat_arr[reInd, hInd, it, i_main] = Dsat
                
                AdjMat = rewire.rewiring_process(AdjMat, actA, re)
                
                [aspA, satA,
                 pcA, actA,
                 cpayA] = Record.update_iterated(pay, aspA,
                                                 satA, payA,
                                                 cpayA, pcA,
                                                 habA, betaA,
                                                 actA, eps,
                                                 totNP)
                
                [CC, CD, DD,
                 Cdeg, Ddeg] = eval_LinksDeg(AdjMat, actA, totNP)
                CD_arr[reInd, hInd, it, i_main] = CD
                CC_arr[reInd, hInd, it, i_main] = CC
                DD_arr[reInd, hInd, it, i_main] = DD
                Cdeg_arr[reInd, hInd, it, i_main] = Cdeg
                Ddeg_arr[reInd, hInd, it, i_main] = Ddeg

        print(time.time()-tTime)
        
    print('re='+str(re)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))

print(time.time()-tTime)


np.savez("heatmap_probchange_reVShab.npz",
         #probs=np.array([0.93, 0.3, 0.87, 0.3, 0.2]),
         re = re_arr, h = h_arr,
         coopfrac = coopfrac_arr, sat = sat_arr,
         Dsat = Dsat_arr, Csat = Csat_arr,
         CC = CC_arr, CD = CD_arr, DD = DD_arr,
         Cdeg = Cdeg_arr, Ddeg = Ddeg_arr)

coopfracheat = np.mean(np.mean(coopfrac_arr[:,:,:,eqbTime:], axis=-1),
                       axis=-1)
ax = sns.heatmap(coopfracheat, vmin=0, vmax=1,
                 cbar_kws={'label': 'Cooperator Fraction'},
                 xticklabels=h_arr,yticklabels=re_arr,
                 cmap='RdBu', fmt="d")
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('coopfracheat_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()

satheat = np.mean(np.mean(sat_arr[:,:,:,eqbTime:], axis=-1),
                       axis=-1)
ax = sns.heatmap(satheat, vmin=np.min(satheat), vmax=np.max(satheat),
                 cbar_kws={'label': 'Average Satisfaction'},
                 xticklabels=h_arr, yticklabels=re_arr,
                 cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('satheat_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()


Csatheat = np.mean(np.mean(Csat_arr[:,:,:,eqbTime:], axis=-1),
                   axis=-1)
Dsatheat = np.mean(np.mean(Dsat_arr[:,:,:,eqbTime:], axis=-1),
                   axis=-1)


satlim = max([np.abs(np.min(Csatheat)), np.abs(np.max(Csatheat)),
               np.abs(np.min(Dsatheat)), np.abs(np.max(Dsatheat))])

ax = sns.heatmap(Csatheat,vmin=-1*satlim, vmax=satlim,
                 cbar_kws={'label':
                           'Average Satisfaction of Cooperators'},
                 xticklabels=h_arr, yticklabels=re_arr,
                 cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('Csatheat_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()


ax = sns.heatmap(Dsatheat,vmin=-1*satlim, vmax=satlim,
                 cbar_kws=
                 {'label':'Average Satisfaction of Defectors'},
                 xticklabels=h_arr, yticklabels=re_arr,
                 cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('Dsatheat_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()

'''
CCmean_arr = np.mean(np.mean(CC_arr, axis=-1), axis=-1)
ax = sns.heatmap(CCmean_arr, cbar_kws={'label':'C-C links'},
                 xticklabels=h_arr, yticklabels=re_arr,
                 cmap='gist_gray')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('CClinks_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()

CDmean_arr = np.mean(np.mean(CD_arr, axis=-1), axis=-1)
ax = sns.heatmap(CDmean_arr, cbar_kws={'label':'C-D links'},
                 xticklabels=h_arr, yticklabels=re_arr,
                 cmap='gist_gray')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('CDlinks_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()

DDmean_arr = np.mean(np.mean(DD_arr, axis=-1), axis=-1)
ax = sns.heatmap(DDmean_arr, cbar_kws={'label':'D-D links'},
                 xticklabels=h_arr, yticklabels=re_arr,
                 cmap='gist_gray')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
plt.xlabel('habituation (h)')
plt.ylabel('rewiring fraction (re)')
plt.savefig('DDlinks_habre_N'+str(totNP)+'_beta'+str(bet)+'.png')
plt.show()
'''
