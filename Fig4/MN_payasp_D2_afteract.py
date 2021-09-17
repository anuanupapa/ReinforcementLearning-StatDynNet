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
import initialize as init
import features
import eval_shift
import rewire

mpl.rc('axes', labelsize=8)
mpl.rc('ytick', labelsize=8)
sns.set(style="white")
sns.set_style("ticks")

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
hab = 0.9
eqbTime = 735
var_arr = beta_arr
Re = 0.0

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
SDaspavg_arr = np.zeros((trials, rounds))
SDpayavg_arr = np.zeros((trials, rounds))
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

            #AdjMat = rewire.rewiring_process(AdjMat, actA, Re)
            
            [aspA, satA,
             pcA, actA,
             cpayA] = Record.update_iterated(pay, aspA,
                                             satA, payA,
                                             cpayA, pcA,
                                             habA, betaA,
                                             actA, eps,
                                             totNP)

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
            SDaspavg_arr[it,i_main-1] = np.sum(
                aspB[sdpos[0]])/(len(sdpos[0])+0.001)
            SDpayavg_arr[it,i_main-1] = np.sum(
                pay[sdpos[0]])/(len(sdpos[0])+0.001)
            
            [coopfrac,
             sais] = Record.measure(actA, satA, AdjMat)
            coopfrac_arr[it, i_main] = coopfrac
            sat_arr[it, i_main] = sais
            # --------------------------------------------
            
    print('beta='+str(bet)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))

    
    plt.clf()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    ax1.plot(trounds[eqbTime:rounds-1], SCasp_arr[0, eqbTime:rounds-1],
             'go--', label='SC asp')
    ax1.plot(trounds[eqbTime:rounds-1], SCpay_arr[0, eqbTime:rounds-1],
             'bo--', label='SC pay')
    ax1.set_xlabel('rounds', fontsize=15)
    ax2.plot(trounds[eqbTime:rounds-1],
             coopfrac_arr[0, eqbTime:rounds-1],
             'co-', label='C-frac', alpha=0.5)
    ax1.legend(loc=(0,1), framealpha=0, ncol=2, fontsize=15)
    ax2.legend(loc=(0.7,1), framealpha=0, fontsize=15)
    #ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    #ax1.set_yticks([0,10000,20000,30000,40000,50000,60000])
    #ax2.set_ylim([0,1.2])
    #ax1.set_ylim([0,75000])
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(left=0.15, bottom=0.13, top=0.9)
    plt.xticks(range(eqbTime, rounds-1, 3))
    plt.savefig("SCAspPay_h"+str(hab)+"_re"+str(Re)+"_D2.tif", dpi=300)
    plt.show()
    

    plt.clf()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    ax1.plot(trounds[eqbTime:rounds-1], SDasp_arr[0, eqbTime:rounds-1],
             'go--', label='SD asp')
    ax1.plot(trounds[eqbTime:rounds-1], SDpay_arr[0, eqbTime:rounds-1],
             'bo--', label='SD pay')
    ax1.set_xlabel('rounds', fontsize=15)
    ax2.plot(trounds[eqbTime:rounds-1],
             coopfrac_arr[0, eqbTime:rounds-1],
             'co-', label='C-frac', alpha=0.5)
    ax1.legend(loc=(0,1), framealpha=0, ncol=2, fontsize=15)
    ax2.legend(loc=(0.7,1), framealpha=0, fontsize=15)
    #ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    #ax1.set_yticks([0,10000,20000,30000,40000,50000,60000])
    #ax2.set_ylim([0,1.2])
    #ax1.set_ylim([0,75000])
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(left=0.15, bottom=0.13, top=0.9)
    plt.xticks(range(eqbTime, rounds-1, 3))
    plt.savefig("SDAspPay_h"+str(hab)+"_re"+str(Re)+"_D2.tif", dpi=300)
    plt.show()


    plt.clf()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    ax1.plot(trounds[eqbTime:rounds-1], UCasp_arr[0, eqbTime:rounds-1],
             'go--', label='UC asp')
    ax1.plot(trounds[eqbTime:rounds-1], UCpay_arr[0, eqbTime:rounds-1],
             'bo--', label='UC pay')
    ax1.set_xlabel('rounds', fontsize=15)
    ax2.plot(trounds[eqbTime:rounds-1], coopfrac_arr[0,
                                                     eqbTime:rounds-1],
             'co-', label='C-frac', alpha=0.5)
    ax2.legend(loc=(0.7,1), framealpha=0, fontsize=15)
    ax1.legend(loc=(0,1), framealpha=0, fontsize=15, ncol=2)
    ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    #ax1.set_yticks([0,10000,20000,30000,40000,50000])
    ax2.set_ylim([0,0.8])
    #ax1.set_ylim([0,60000])
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(left=0.15, bottom=0.13, top=0.9)
    plt.xticks(range(eqbTime, rounds-1, 3))
    plt.savefig("UCAspPay_h"+str(hab)+"_re"+str(Re)+"_D2.tif", dpi=300)
    plt.show()


    plt.clf()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    ax1.plot(trounds[eqbTime:rounds-1], UDasp_arr[0, eqbTime:rounds-1],
             'go--', label='UD asp')
    ax1.plot(trounds[eqbTime:rounds-1], UDpay_arr[0, eqbTime:rounds-1],
             'bo--', label='UD pay')
    ax1.set_xlabel('rounds', fontsize=15)
    ax1.legend(loc=(0.8,1))
    ax2.plot(trounds[eqbTime:rounds-1], coopfrac_arr[0,
                                                     eqbTime:rounds-1],
             'co-', label='C-frac', alpha=0.3)
    ax2.legend(loc=(0.7,1), framealpha=0, fontsize=15)
    ax1.legend(loc=(0,1), framealpha=0, fontsize=15, ncol=2)
    ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    #ax1.set_yticks([0,10000,20000,30000,40000])
    ax2.set_ylim([0,0.8])
    #ax1.set_ylim([0,50000])
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(left=0.15, bottom=0.13, top=0.9)    
    plt.xticks(range(eqbTime, rounds-1, 3))
    plt.savefig("UDAspPay_h"+str(hab)+"_re"+str(Re)+"_D2.tif", dpi=300)
    plt.show()



    
    plt.clf()
    f, (ax1, ax3) = plt.subplots(2,1, sharex=True)
    ax2 = ax1.twinx()
    ax1.plot(trounds[eqbTime:rounds-1], SDasp_arr[0, eqbTime:rounds-1],
             'go--', label='SD asp')
    ax1.plot(trounds[eqbTime:rounds-1], SDpay_arr[0, eqbTime:rounds-1],
             'bo--', label='SD pay')
    ax2.plot(trounds[eqbTime:rounds-1], coopfrac_arr[0,
                                                     eqbTime:rounds-1],
             'co-', label='C-frac', alpha=0.5)
    ax1.legend(loc=(-0.2,0.98), framealpha=0, ncol=2, fontsize=15)
    ax2.legend(loc=(0.7,0.98), framealpha=0, fontsize=15)
    ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax4=ax3.twinx()
    ax3.plot(trounds[eqbTime:rounds-1], UCasp_arr[0, eqbTime:rounds-1],
             'go--', label='UC asp')
    ax3.plot(trounds[eqbTime:rounds-1], UCpay_arr[0, eqbTime:rounds-1],
             'bo--', label='UC pay')
    ax3.set_xlabel('rounds', fontsize=15)
    ax3.legend(loc=(0.8,1))
    ax4.plot(trounds[eqbTime:rounds-1], coopfrac_arr[0,
                                                     eqbTime:rounds-1],
             'co-', label='C-frac', alpha=0.3)
    ax4.legend(loc=(0.7,0.98), framealpha=0, fontsize=15)
    ax3.legend(loc=(-0.2,0.98), framealpha=0, fontsize=15, ncol=2)
    ax4.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    ax3.tick_params(axis='y', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(left=0.15, bottom=0.13, top=0.92)    
    plt.subplots_adjust(left=0.15,
                        bottom=0.13, 
                        top=0.94, 
                        wspace=0.1, 
                        hspace=0.3)
    plt.savefig("SD&UCAspPay_h"+
                str(hab)+"_re"+str(Re)+"_D2.tif", dpi=300)
    plt.show()

    
[SCtSD_arr, SCtUC_arr, SCtUD_arr,
 SDtSC_arr, SDtUC_arr, SDtUD_arr,
 UCtSC_arr, UCtSD_arr, UCtUD_arr,
 UDtSC_arr, UDtSD_arr, UDtUC_arr,
 SCtSC_arr, SDtSD_arr,
 UCtUC_arr, UDtUD_arr] = eval_shift.ClassShift(category_arr)



plt.clf()

ax1  = plt.axes()
ax2 = ax1.twinx()
ax2.plot(trounds[eqbTime:rounds-1], coopfrac_arr[0,eqbTime:rounds-1],
         'o-', label='C-frac', alpha=0.65, c='c')
ax1.plot(trounds[eqbTime:rounds-1], SDtUC_arr[0,0,eqbTime:rounds-1],
         'o-', label=r'SD$\Rightarrow$UC', c='green')
ax1.plot(trounds[eqbTime:rounds-1], UCtUD_arr[0,0,eqbTime:rounds-1],
         'o-', label=r'UC$\Rightarrow$UD', c='peru')
ax1.plot(trounds[eqbTime:rounds-1], UDtSD_arr[0,0,eqbTime:rounds-1],
         'o-', label=r'UD$\Rightarrow$SD', c='red')
ax1.set_xlabel('rounds', fontsize=15)
ax2.legend(loc=(0.83,0.98), framealpha=0, fontsize=15)
ax1.legend(loc=(-0.2,1.04), framealpha=0, fontsize=15, ncol=3)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.set_ylabel('dominant transitions', fontsize=15)
#ax1.set_ylim(0,450)
#ax2.set_ylim(0,0.6)
plt.subplots_adjust(left=0.15, bottom=0.13, top=0.9)   
plt.xticks(range(eqbTime, rounds-1, 3))
plt.savefig('DominantTransitions_h'+str(hab)+'_re'+str(Re)+'D2.tif'
            ,dpi=300)
plt.show()


plt.clf()
ax1 = plt.axes()
ax2 = ax1.twinx()
ax1.plot(trounds[eqbTime:rounds-1], UCtUD_arr[0,0,eqbTime:rounds-1],
         'o-', label=r'UC$\Rightarrow$UD', c='red', alpha=1)
ax2.plot(trounds[eqbTime:rounds-1], UCtSC_arr[0,0,eqbTime:rounds-1],
         'o-', label=r'UC$\Rightarrow$SC', c='darkblue', alpha=1)
ax2.set_xlabel('rounds')
ax1.legend(loc=(0.0,.98), prop={'size': 15}, framealpha=0)
ax2.legend(loc=(0.7,0.98), prop={'size': 15}, framealpha=0)
plt.xlabel('rounds')
plt.savefig("BifurcationTransition_D2.tif", dpi=300)
plt.show()


np.savez("D2_arrays_beta"
         +str(bet)+"_hab"+str(hab)+".npz",

         cfrac = coopfrac_arr, rounds = trounds,
         
         SC=SC_arr, SD=SD_arr,
         UC=UC_arr, UD=UD_arr,

         SCpay = SCpay_arr, SCasp = SCasp_arr,
         UCpay = UCpay_arr, UCasp = UCasp_arr,
         SDpay = SDpay_arr, SDasp = SDasp_arr,
         UDpay = UDpay_arr, UDasp = UDasp_arr,
         
         SCtSC=SCtSC_arr, SCtSD=SCtSD_arr,
         SCtUC=SCtUC_arr, SCtUD=SCtUD_arr,
         
         SDtSC=SDtSC_arr, SDtSD=SDtSD_arr,
         SDtUC=SDtUC_arr, SDtUD=SDtUD_arr,
         
         UCtSC=UCtSC_arr, UCtSD=UCtSD_arr,
         UCtUC=UCtUC_arr, UCtUD=UCtUD_arr,
         
         UDtSC=UDtSC_arr, UDtSD=UDtSD_arr,
         UDtUC=UDtUC_arr, UDtUD=UDtUD_arr)
