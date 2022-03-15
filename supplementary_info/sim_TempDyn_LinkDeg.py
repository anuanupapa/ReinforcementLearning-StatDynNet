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
hab = 0.4
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

            '''
            # A player is satisfied and decides an action
            [SC,UC,SD,UD,category,
             SCpos, UCpos,
             SDpos, UDpos] = features.feature(satA, actA)
            category_arr[varInd, it, i_main, :] = category
            SC_arr[it, i_main] = SC/totNP
            UC_arr[it, i_main] = UC/totNP
            SD_arr[it, i_main] = SD/totNP
            UD_arr[it, i_main] = UD/totNP
            '''
            
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

    cutoff = 500
    
    x = np.linspace(min(coopfrac_arr*totNP),
                    max(coopfrac_arr)*totNP, 100)
    x1 = min(coopfrac_arr*totNP)
    x2 = max(coopfrac_arr*totNP)
    fig, ax = plt.subplots(3,1)
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    fitC_params=np.polyfit(np.log(coopfrac_arr[0,cutoff:]*totNP),
                           np.log(CC_arr[0,cutoff:]), 1)
    print(fitC_params)
    ax1.plot(coopfrac_arr[0,cutoff:]*totNP, CC_arr[0,cutoff:], 'bo',
             label=r'slope$\approx$'+str(np.round(fitC_params[0],3)))
    ax1.plot(x1, x1**fitC_params[0]*(np.exp(fitC_params[1])),
             x2, x2**fitC_params[0]*(np.exp(fitC_params[1])), 'b')
    ax1.legend(framealpha=0)
    ax1.set_ylabel("C-C Links")
    ax1.set_xlabel(r'$N_{C}$')
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_label_coords(0.52, -0.08)

    fitD_params=np.polyfit(np.log((1-coopfrac_arr[0,cutoff:])*totNP), 
                           np.log(DD_arr[0,cutoff:]), 1)
    print(fitD_params)
    ax2.plot((1-coopfrac_arr[0,cutoff:])*totNP, DD_arr[0,cutoff:],'ro',
             label=r'slope$\approx$'+str(np.round(fitD_params[0],3)))
    ax2.plot(x1, x1**fitD_params[0]*(np.exp(fitD_params[1])),
             x2, x2**fitD_params[0]*(np.exp(fitD_params[1])), 'r')
    ax2.legend(framealpha=0)
    ax2.set_ylabel("D-D Links")
    ax2.set_xlabel(r'$N_{D}$')
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.xaxis.set_label_coords(0.52, -0.08)

    x1 = min(coopfrac_arr*(1-coopfrac_arr)*(totNP**2))
    x2 = max(coopfrac_arr*(1-coopfrac_arr)*(totNP**2))
    fitCD_params=np.polyfit(
        np.log((1-coopfrac_arr[0,cutoff:
                               ])*coopfrac_arr[0,cutoff:]*(totNP**2)),
        np.log(CD_arr[0,cutoff:]), 1)
    print(fitCD_params)    
    ax3.plot(
        (1-coopfrac_arr[0,cutoff:])*coopfrac_arr[0,cutoff:]*(totNP**2),
        CD_arr[0,cutoff:], 'ko',
        label=r'slope$\approx$'+str(np.round(fitCD_params[0],3)))
    ax3.plot(x1, x1**fitCD_params[0]*(np.exp(fitCD_params[1])),
             x2, x2**fitCD_params[0]*(np.exp(fitCD_params[1])), 'k')
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.legend(framealpha=0)
    ax3.set_ylabel("C-D Links")
    ax3.set_xlabel(r'$N_{C}*N_{D}$')
    ax3.xaxis.set_label_coords(0.52, -0.08)
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.12, 
                        right=0.98, 
                        top=0.95, 
                        wspace=0.1, 
                        hspace=0.5)
    plt.savefig("S3-LinkScaling.jpg")
    plt.savefig("S3-LinkScaling.png")
    plt.show()
