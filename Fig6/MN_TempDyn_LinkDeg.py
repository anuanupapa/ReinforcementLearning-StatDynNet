import numpy as np
import matplotlib.pyplot as plt
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
#import windowing

sns.set(style="whitegrid")
sns.set_style("ticks")

# Double counting
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
eqbTime = 700
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

            # A player is satisfied and decides an action
            [SC,UC,SD,UD,category,
             SCpos, UCpos,
             SDpos, UDpos] = features.feature(satA, actA)
            category_arr[varInd, it, i_main, :] = category
            SC_arr[it, i_main] = SC/totNP
            UC_arr[it, i_main] = UC/totNP
            SD_arr[it, i_main] = SD/totNP
            UD_arr[it, i_main] = UD/totNP

            [coopfrac,
             sais] = Record.measure(actA, satA, AdjMat)
            coopfrac_arr[it, i_main] = coopfrac
            sat_arr[it, i_main] = sais

            [C_C, C_D, D_D, Cdeg, Ddeg
             ] = eval_LinksDeg(AdjMat, actA, totNP)
            Cdeg_arr[it, i_main] = Cdeg
            Ddeg_arr[it, i_main] = Ddeg
            CC_arr[it, i_main] = C_C#/(C_C + D_D + C_D)
            DD_arr[it, i_main] = D_D#/(C_C + D_D + C_D)
            CD_arr[it, i_main] = C_D#/(C_C + D_D + C_D)

            
    print('beta='+str(bet)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))

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
    plt.subplots_adjust(right=.9, left=0.15, bottom=0.13)
    #plt.savefig("Deg_reNet_re"+str(Re)+"_h"+str(hab)+".png")
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
    plt.savefig("Links_reNet_re"+str(Re)+"_h"+str(hab)+".tif")
    plt.show()

np.savez("LinkDeg_arrays_beta"
         +str(bet)+"_hab"+str(hab)+"_re"+str(Re)+".npz",

         cfrac = coopfrac_arr, rounds = trounds,
         
         SC=SC_arr, SD=SD_arr,
         UC=UC_arr, UD=UD_arr,

         CC=CC_arr, CD=CD_arr, DD=DD_arr,
         Cdeg=Cdeg_arr, Ddeg=Ddeg_arr)
