import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as scsp
import networkx as nx
import time
from datetime import datetime
import numba as nb
from numba import int64, float64
import seaborn as sns
import PGG
import Record_vectorized as Record
import rewire_sat as rewire
import initialize as init
import features
import eval_shift
#import windowing

sns.set(style="white")
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
    C_deg = C_deg/(C_count + 0.001)
    D_deg = D_deg/(N - C_count + 0.001)
    return(CC/2., CD/2., DD/2., C_deg, D_deg)


tTime = time.time()
totTime = 0.

totNP = 500
trials = 1
rounds = 750
trounds = np.arange(0,rounds,1)

eps = 0.02
c = 1.
r = 2
b = r*c
ini_re = 0.3
beta_arr = np.array([10**(0.)])/c
hab = 0.1
re = 0.3
eqbTime = 650
var_arr = beta_arr

p1, p0 = 0.8, 0.2
probabs = np.array([p1, p0])

print(len(beta_arr))

coopfrac_arr = np.zeros((trials, rounds))
sat_arr = np.zeros((trials, rounds))
avgpay_arr = np.zeros((trials, rounds))
avgasp_arr = np.zeros((trials, rounds))
SC_arr = np.zeros((trials, rounds))
UC_arr = np.zeros((trials, rounds))
SD_arr = np.zeros((trials, rounds))
UD_arr = np.zeros((trials, rounds))
satSC_arr = np.zeros((trials, rounds))
category_arr = np.zeros((len(var_arr), trials, rounds, totNP, 1))
SCasp_arr = np.zeros((trials, rounds))
UCasp_arr = np.zeros((trials, rounds))
SDasp_arr = np.zeros((trials, rounds))
UDasp_arr = np.zeros((trials, rounds))
SCpay_arr = np.zeros((trials, rounds))
UCpay_arr = np.zeros((trials, rounds))
SDpay_arr = np.zeros((trials, rounds))
UDpay_arr = np.zeros((trials, rounds))
Cdeg_arr = np.zeros((trials, rounds))
Ddeg_arr = np.zeros((trials, rounds))
CC_arr = np.zeros((trials, rounds))
DD_arr = np.zeros((trials, rounds))
CD_arr = np.zeros((trials, rounds))
category = np.zeros((totNP, 1))

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

            avgasp_arr[it, i_main] = np.mean(aspA)
            avgpay_arr[it, i_main] = np.mean(pay)
            aspB = np.copy(aspA)

            # The asp  will be used to determine sat in next round
            [aspA, satA,
             pcA, actA,
             cpayA] = Record.update_iterated(pay, aspA,
                                             satA, payA,
                                             cpayA, pcA,
                                             habA, betaA,
                                             actA, eps,
                                             totNP)

            AdjMat = rewire.rewiring_process(AdjMat, satA, re, probabs)

            # Following  is just recording ------------------
            # A player is satisfied and decides an action
            categoryB = category.copy()
            [SC,UC,SD,UD,category,
             SCpos, UCpos,
             SDpos, UDpos] = features.feature(satA, actA)
            
            category_arr[varInd, it, i_main-1, :] = categoryB
            #categoryB = category.copy()
            SC_arr[it, i_main] = SC/totNP
            UC_arr[it, i_main] = UC/totNP
            SD_arr[it, i_main] = SD/totNP
            UD_arr[it, i_main] = UD/totNP
            scpos = np.where(categoryB == 4)
            ucpos = np.where(categoryB == 2)
            sdpos = np.where(categoryB == -2)
            udpos = np.where(categoryB == -4)
            #Aspiration before the update is done
            SCasp_arr[it, i_main-1] = np.sum(aspB[scpos[0]])
            UCasp_arr[it, i_main-1] = np.sum(aspB[ucpos[0]])
            SDasp_arr[it, i_main-1] = np.sum(aspB[sdpos[0]])
            UDasp_arr[it, i_main-1] = np.sum(aspB[udpos[0]])
            SCpay_arr[it, i_main-1] = np.sum(pay[scpos[0]])
            UCpay_arr[it, i_main-1] = np.sum(pay[ucpos[0]])
            SDpay_arr[it, i_main-1] = np.sum(pay[sdpos[0]])
            UDpay_arr[it, i_main-1] = np.sum(pay[udpos[0]])
            
            [coopfrac,
             sais] = Record.measure(actA, satA, AdjMat)
            coopfrac_arr[it, i_main] = coopfrac
            sat_arr[it, i_main] = sais
            avgasp_arr[it, i_main-1] = np.sum(aspB)/totNP
            avgpay_arr[it, i_main-1] = np.sum(pay)/totNP

            [C_C, C_D, D_D, Cdeg, Ddeg
             ] = eval_LinksDeg(AdjMat, actA, totNP)
            Cdeg_arr[it, i_main] = Cdeg
            Ddeg_arr[it, i_main] = Ddeg
            CC_arr[it, i_main] = C_C
            DD_arr[it, i_main] = D_D
            CD_arr[it, i_main] = C_D
            
            # --------------------------------------------
            
    print('beta='+str(bet)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))
    print(np.mean(coopfrac_arr[0, 400:rounds]))
    
    '''
    plt.plot(trounds[eqbTime:], SC_arr[0,eqbTime:], 'bo-', label='SC')
    plt.plot(trounds[eqbTime:], UC_arr[0,eqbTime:], 'go-', label='UC')
    plt.plot(trounds[eqbTime:], SD_arr[0,eqbTime:], 'ro-', label='SD')
    plt.plot(trounds[eqbTime:], UD_arr[0,eqbTime:], 'o-', label='UD',
             color='peru')
    plt.legend(loc=(0,0.98), fontsize=15, framealpha=0, ncol=4)
    plt.xlabel('rounds', fontsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(bottom=0.13, right=0.85)
    plt.savefig('SatAct_LI_resat'+str(re)+'_hab'+str(hab)+'.png')
    #plt.savefig('LinkPayAspSatAct_LI_resat'
    #+str(re)+'_hab'+str(hab)+'.png')
    plt.show()
    
    
    ax3 = plt.axes()
    ax2 = ax3.twinx()
    ax3.plot(trounds[eqbTime:rounds-1],
             coopfrac_arr[0,eqbTime:rounds-1],
             'co-', label='C-frac')
    ax2.plot(trounds[eqbTime:rounds-1], CC_arr[0,eqbTime:rounds-1],
             'bo-', label='C-C', alpha=0.5)
    ax2.plot(trounds[eqbTime:rounds-1], CD_arr[0,eqbTime:rounds-1],
             'ko-', label='C-D', alpha=0.5)
    ax2.plot(trounds[eqbTime:rounds-1], DD_arr[0,eqbTime:rounds-1],
             'ro-', label='D-D', alpha=0.5)
    ax3.set_xlabel('rounds', fontsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.legend(loc=(-0.1,.98), framealpha=0, fontsize=15, ncol=1)
    ax2.legend(loc=(0.5, 0.98), framealpha=0, fontsize=15, ncol=3)
    plt.subplots_adjust(bottom=0.13, right=0.85)
    #ax2.set_ylim(0,49999)
    plt.savefig('Links_LI_resat'+str(re)+'_hab'+str(hab)+'.png')
    plt.show()
    '''
    
    fig, ax = plt.subplots(3, 1, sharex=True)
    #ax1 = plt.axes()
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]

    ax1.plot(trounds[eqbTime:], SC_arr[0,eqbTime:], 'bo-', label='SC')
    ax1.plot(trounds[eqbTime:], UC_arr[0,eqbTime:], 'go-', label='UC')
    ax1.plot(trounds[eqbTime:], SD_arr[0,eqbTime:], 'o-', label='SD',
             color='red')
    ax1.plot(trounds[eqbTime:], UD_arr[0,eqbTime:], 'o-', label='UD',
             color='peru')
    ax1.legend(loc=(0,0.92), fontsize=15, framealpha=0, ncol=4)
    ax1.set_ylim(0,1)
    
    ax2.plot(trounds[eqbTime:rounds-1], avgasp_arr[0,eqbTime:rounds-1],
             'g*--', label='<asp>')
    ax2.plot(trounds[eqbTime:rounds-1], avgpay_arr[0,eqbTime:rounds-1],
             'b*--', label='<pay>')
    ax2.legend(loc=(0,0.92), framealpha=0, fontsize=15, ncol=2)

    ax3.plot(trounds[eqbTime:rounds-1], CC_arr[0,eqbTime:rounds-1],
             'o-',c='c', label='C-C')
    ax3.plot(trounds[eqbTime:rounds-1], CD_arr[0,eqbTime:rounds-1],
             'ko-', label='C-D')
    ax3.plot(trounds[eqbTime:rounds-1], DD_arr[0,eqbTime:rounds-1],
             'ro-', label='D-D')
    ax3.legend(loc=(0,0.92), framealpha=0, fontsize=15, ncol=3)

    ax3.set_xlabel('rounds', fontsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.set_xticks([650, 675, 700, 725, 750])
    
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    
    ax1.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(left=0.14,
                        bottom=0.12, 
                        right=0.99,
                        top=0.93, 
                        wspace=0.1, 
                        hspace=0.35)
    #plt.savefig('LinkPayAspSatAct_LI_resat'
     #           +str(re)+'_hab'+str(hab)+'.tif', dpi=300)
    plt.show()

'''
np.savez("LI_LinkDegSatactPayAsp_arrays_beta"
         +str(bet)+"_hab"+str(hab)+"_re"+str(re)+".npz",

         cfrac = coopfrac_arr, rounds = trounds,
         
         SC=SC_arr, SD=SD_arr,
         UC=UC_arr, UD=UD_arr,

         avgasp=avgasp_arr, avgpay=avgpay_arr,

         CC=CC_arr, CD=CD_arr, DD=DD_arr,
         Cdeg=Cdeg_arr, Ddeg=Ddeg_arr)
'''
