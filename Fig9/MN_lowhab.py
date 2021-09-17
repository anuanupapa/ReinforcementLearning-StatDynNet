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
rounds = 550
eqbTime = 440
trounds = np.arange(0,rounds,1)

eps = 0.02
c = 1.
r = 2.
b = r*c
ini_re = 0.3
beta_arr = np.array([10**(0.)])/c
hab = 0.1
var_arr = beta_arr
Re = 1.0
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

category_arr = np.zeros((len(var_arr), trials, rounds, totNP, 1))

aspHist=[]
aspHist_round=[]
varInd = -1
counter1 = 0
counter2 = 0
counter3 = 0
payHist=[]
payHist_round=[]
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
        SD,UD=0,0
        for i_main in range(rounds):

            print(i_main)
            pay = PGG.game(AdjMat, actA, r, c, totNP)

            aspB = np.copy(aspA)
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

            #print(np.max(aspA), np.min(aspA))
            
            if UC_arr[it, i_main]>0.9 and i_main>539 and counter1<3:
                aspHist.append(aspB)
                print(i_main)
                aspHist_round.append(i_main)
                counter1 = counter1 + 1
                payHist.append(pay)
            else:
                pass

            if SC_arr[it, i_main]>0.3 and i_main>539 and counter2<3:
                aspHist.append(aspB)
                print(i_main)
                aspHist_round.append(i_main)
                counter2 = counter2 + 1
                payHist.append(pay)
            else:
                pass
            
            if SD_arr[it, i_main]>0.3 and i_main>539 and counter3<3:
                aspHist.append(aspB)
                print(i_main)
                aspHist_round.append(i_main)
                counter3 = counter3 + 1
                payHist.append(pay)
            else:
                pass
                        
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

    f, ax = plt.subplots(2, 3, sharey='row')

    ax1, ax2, ax3, ax4, ax5, ax6 = ax[0,0], ax[1,0], ax[0,1], ax[1,1], ax[0,2], ax[1,2], 
    ax1.hist(aspHist[0], bins=15,
             label="round "+str(aspHist_round[0]), color='teal',
             alpha=1.)
    ax1.hist(aspHist[1], bins=15 ,
             label="round "+str(aspHist_round[1]), color='maroon',
             alpha=0.7)
    ax1.hist(aspHist[2], bins=15 ,
             label="round "+str(aspHist_round[2]), color='forestgreen',
             alpha=0.4)
    ax1.set_xlim(0,250)
    
    ax3.hist(aspHist[3], bins=15 ,
             label="round "+str(aspHist_round[3]), color='teal',
             alpha=1)
    ax3.hist(aspHist[4], bins=15 ,
             label="round "+str(aspHist_round[4]), color='maroon',
             alpha=0.7)
    ax3.hist(aspHist[5], bins=15 ,
             label="round "+str(aspHist_round[5]), color='forestgreen',
             alpha=0.4)
    ax3.set_xlim(0,250)
    
    ax5.hist(aspHist[6], bins=15 ,
             label="round "+str(aspHist_round[6]), color='teal',
             alpha=1.)
    ax5.hist(aspHist[7], bins=15 ,
             label="round "+str(aspHist_round[7]), color='maroon',
             alpha=0.7)
    ax5.hist(aspHist[8], bins=15 ,
             label="round "+str(aspHist_round[8]), color='forestgreen',
             alpha=0.4)
    ax5.set_xlim(0,250)
    
    ax2.hist(payHist[0], bins=15,
             label="round "+str(aspHist_round[0]), color='teal',
             alpha=1., edgecolor='k')
    ax2.hist(payHist[1], bins=15,
             label="round "+str(aspHist_round[1]), color='maroon',
             alpha=0.7, edgecolor='k')
    ax2.hist(payHist[2], bins=15,
             label="round "+str(aspHist_round[2]), color='forestgreen',
             alpha=0.4, edgecolor='k')
    ax2.set_xlim(-100,500)
    
    ax4.hist(payHist[3], bins=15,
             label="round "+str(aspHist_round[3]), color='teal',
             alpha=1, edgecolor='k')
    ax4.hist(payHist[4], bins=15,
             label="round "+str(aspHist_round[4]), color='maroon',
             alpha=0.7, edgecolor='k')
    ax4.hist(payHist[5], bins=15,
             label="round "+str(aspHist_round[5]), color='forestgreen',
             alpha=0.4, edgecolor='k')
    ax4.set_xlim(-100,500)    
    
    ax6.hist(payHist[6], bins=15,
             label="round "+str(aspHist_round[6]), color='teal',
             alpha=1., edgecolor='k')
    ax6.hist(payHist[7], bins=15,
             label="round "+str(aspHist_round[7]), color='maroon',
             alpha=0.7, edgecolor='k')
    ax6.hist(payHist[8], bins=15,
             label="round "+str(aspHist_round[8]), color='forestgreen',
             alpha=0.4, edgecolor='k')
    ax6.set_xlim(-100,500)
    
    ax4.set_xlabel("payoff", fontsize=15)
    ax2.set_ylabel("number of players", fontsize=15)
    ax1.legend(loc=(0.05, 0.98), framealpha=0)
    ax3.legend(loc=(0.05, 0.98), framealpha=0)
    ax5.legend(loc=(0.05, 0.98), framealpha=0)
    '''
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)
    ax4.tick_params(axis='x', labelsize=15)
    ax5.tick_params(axis='y', labelsize=15)
    ax5.tick_params(axis='x', labelsize=15)
    ax6.tick_params(axis='y', labelsize=15)
    ax6.tick_params(axis='x', labelsize=15)
    '''
    ax2.yaxis.set_label_coords(-0.24, 1.2)
    ax3.set_xlabel("aspiration", fontsize=15)
    plt.subplots_adjust(left=0.1,
                        bottom=0.12, 
                        right=0.98, 
                        top=0.85, 
                        wspace=0.1, 
                        hspace=0.4)
    plt.savefig("histogram_lowhab.tif", dpi=300)
    plt.show()

    
    

    
    plt.clf()
    ax1 = plt.axes()
    ax1.plot(trounds[eqbTime+100:], SC_arr[0,eqbTime+100:],
             'bo-', label='SC')
    ax1.plot(trounds[eqbTime+100:], UC_arr[0,eqbTime+100:],
             'go-', label='UC')
    ax1.plot(trounds[eqbTime+100:], SD_arr[0,eqbTime+100:],
             'o-', label='SD', color='peru')
    ax1.plot(trounds[eqbTime+100:], UD_arr[0,eqbTime+100:],
             'ro-', label='UD')
    ax1.legend(loc=(-0.1,1), fontsize=15, framealpha=0, ncol=4)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.grid(False)
    ax1.set_xlabel("round", fontsize=15)
    ax1.set_ylabel("fraction of players in each category", fontsize=15)
    plt.subplots_adjust(right=0.9, left=0.15, bottom=0.13, top=0.9)
    plt.savefig("Satact_polarization.tif", dpi=300)
    plt.show()    


    ax2 = plt.axes()
    ax1 = ax2.twinx()
    ax1.plot(trounds[eqbTime+100:],
             coopfrac_arr[0,eqbTime+100:],
             'co-', linewidth=1, label='C-frac')
    ax2.plot(trounds[eqbTime+100:], Cdeg_arr[0,eqbTime+100:], 'bo-',
             label='<deg(C)>')
    ax2.plot(trounds[eqbTime+100:], Ddeg_arr[0,eqbTime+100:], 'ro-',
             label='<deg(D)>')
    ax1.legend(loc=(0.7,0.98), fontsize=15, framealpha=0)
    ax2.legend(loc=(0.0,0.98), fontsize=15, framealpha=0, ncol=1)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax1.grid(False)
    ax2.grid(False)
    ax2.set_xlabel("round", fontsize=15)
    plt.subplots_adjust(right=.9, left=0.15, bottom=0.13)
    #plt.savefig("Deg_reNet_re"+str(Re)+"_h"+str(hab)+".tif")
    #plt.show()
    

    plt.clf()
    ax2 = plt.axes()
    plt.xlabel('rounds')
    ax2.plot(trounds[eqbTime+100:], CC_arr[0,eqbTime+100:], 'bo-',
             label='C-C')
    ax2.plot(trounds[eqbTime+100:], DD_arr[0,eqbTime+100:], 'ro-',
             label='D-D')
    ax2.plot(trounds[eqbTime+100:], CD_arr[0,eqbTime+100:], 'ko-',
             label='C-D')
    ax2.legend(loc=(-0.1,1.0), fontsize=15, framealpha=0, ncol=3)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    #ax2.set_ylabel("total number of links")
    ax2.grid(False)
    ax2.set_xlabel("round", fontsize=15)
    plt.subplots_adjust(right=0.9, left=0.15, bottom=0.13, top=0.9)
    plt.savefig("Links_HI_re"+str(Re)+"_h"+str(hab)+".tif", dpi=300)
    plt.show()



    
    
    '''
    plt.clf()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    ax1.plot(trounds[eqbTime:], SCasp_arr[0, eqbTime:],
             'go-', label='SC asp')
    ax1.plot(trounds[eqbTime:], SCpay_arr[0, eqbTime:],
             'bo-', label='SC pay')
    ax1.set_xlabel('rounds', fontsize=15)
    ax2.plot(trounds[eqbTime:], coopfrac_arr[0, eqbTime:],
             'co-', label='c-frac', alpha=0.5)
    ax1.legend(loc=(0,0.98), framealpha=0, ncol=1, fontsize=15)
    ax2.legend(loc=(0.7,0.98), framealpha=0, fontsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    plt.subplots_adjust(right=0.9, left=0.15, bottom=0.13)
    plt.savefig("SCAspPay_h"+str(hab)+"_re"+str(Re)+"_D1.tif")
    plt.show()
    '''


[SCtSD_arr, SCtUC_arr, SCtUD_arr,
 SDtSC_arr, SDtUC_arr, SDtUD_arr,
 UCtSC_arr, UCtSD_arr, UCtUD_arr,
 UDtSC_arr, UDtSD_arr, UDtUC_arr,
 SCtSC_arr, SDtSD_arr,
 UCtUC_arr, UDtUD_arr] = eval_shift.ClassShift(category_arr)

'''
plt.clf()
fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(trounds[eqbTime+100:], SDtUC_arr[0,0,eqbTime+100:],
         'o-', label=r'SD$\Rightarrow$UC', c='green')
ax1.plot(trounds[eqbTime+100:], UCtUD_arr[0,0,eqbTime+100:],
         'o-', label=r'UC$\Rightarrow$UD', c='peru')
ax1.plot(trounds[eqbTime+100:], UDtSD_arr[0,0,eqbTime+100:],
         'o-', label=r'UD$\Rightarrow$SD', c='red')
ax2.plot(trounds[eqbTime+100:], SCtUD_arr[0,0,eqbTime+100:],
         'o-', label=r'SC$\Rightarrow$UD', c='peru')
ax2.plot(trounds[eqbTime+100:], UDtUC_arr[0,0,eqbTime+100:],
         'o-', label=r'UD$\Rightarrow$UC', c='green')
ax2.plot(trounds[eqbTime+100:], UCtSC_arr[0,0,eqbTime+100:],
         'o-', label=r'UC$\Rightarrow$SC', c='blue')
ax2.set_xlabel('rounds', fontsize=15)
ax1.legend(loc=(-0.1,0.95), prop={'size': 15}, ncol=3, framealpha=0)
ax2.legend(loc=(-0.1,0.95), prop={'size': 15}, ncol=3, framealpha=0)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax2.tick_params(axis='x', labelsize=15)
plt.xticks(fontsize=15)
plt.subplots_adjust(right=0.9, left=0.15, bottom=0.13, top=0.9)
plt.savefig("Transitions_cyc1&2_HI_h"+str(hab)+"_re"+str(Re)+".tif",
            dpi=300)
plt.show()




#SD UC UD SD
plt.clf()
ax1 = plt.axes()
ax1.plot(trounds[eqbTime+100:], coopfrac_arr[0,eqbTime+100:],
         'o-', label='C-frac', alpha=0.3,
         c='blue')
ax2 = ax1.twinx()
ax2.plot(trounds[eqbTime+100:], SDtUC_arr[0,0,eqbTime+100:],
         'o-', label=r'SD$\Rightarrow$UC', c='green')
ax2.plot(trounds[eqbTime+100:], UCtUD_arr[0,0,eqbTime+100:],
         'o-', label=r'UC$\Rightarrow$UD', c='peru')
ax2.plot(trounds[eqbTime+100:], UDtSD_arr[0,0,eqbTime+100:],
         'o-', label=r'UD$\Rightarrow$SD', c='red')
ax2.set_xlabel('rounds', fontsize=15)
ax1.legend(loc=(0,0.98), prop={'size': 15}, framealpha=0)
ax2.legend(loc=(0.4,0.98), prop={'size': 15}, ncol=2, framealpha=0)
plt.xlabel('rounds', fontsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
#plt.savefig("Transitions_cyc2_HI_h"+str(hab)+"_re"+str(Re)+".tif")
plt.show()


#SC UC UD SC
plt.clf()
ax1 = plt.axes()
ax1.plot(trounds[eqbTime+100:], coopfrac_arr[0,eqbTime+100:],
         'o-', label='C-frac', alpha=0.3,
         c='blue')
ax2 = ax1.twinx()
ax2.plot(trounds[eqbTime+100:], SCtUD_arr[0,0,eqbTime+100:],
         'o-', label=r'SC$\Rightarrow$UD', c='peru')
ax2.plot(trounds[eqbTime+100:], UDtUC_arr[0,0,eqbTime+100:],
         'o-', label=r'UD$\Rightarrow$UC', c='green')
ax2.plot(trounds[eqbTime+100:], UCtSC_arr[0,0,eqbTime+100:],
         'o-', label=r'UC$\Rightarrow$SC', c='blue')
ax2.set_xlabel('rounds', fontsize=15)
ax1.legend(loc=(0,0.98), prop={'size': 15}, framealpha=0)
ax2.legend(loc=(0.4,0.98), prop={'size': 15}, ncol=2, framealpha=0)
plt.xlabel('rounds', fontsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
#plt.savefig("Transitions_cyc1_HI_h"+str(hab)+"_re"+str(Re)+".tif")
plt.show()
'''
np.savez("LinkDegSatactPayAsp_arrays_beta"
         +str(bet)+"_hab"+str(hab)+"_re"+str(Re)+".npz",

         cfrac = coopfrac_arr, rounds = trounds,
         
         SC=SC_arr, SD=SD_arr,
         UC=UC_arr, UD=UD_arr,

         asphist=aspHist, payhist=payHist,
         
         CC=CC_arr, CD=CD_arr, DD=DD_arr,
         Cdeg=Cdeg_arr, Ddeg=Ddeg_arr,
         
         SCtSC=SCtSC_arr, SCtSD=SCtSD_arr,
         SCtUC=SCtUC_arr, SCtUD=SCtUD_arr,
         
         SDtSC=SDtSC_arr, SDtSD=SDtSD_arr,
         SDtUC=SDtUC_arr, SDtUD=SDtUD_arr,
         
         UCtSC=UCtSC_arr, UCtSD=UCtSD_arr,
         UCtUC=UCtUC_arr, UCtUD=UCtUD_arr,
         
         UDtSC=UDtSC_arr, UDtSD=UDtSD_arr,
         UDtUC=UDtUC_arr, UDtUD=UDtUD_arr)

plt.clf()
fig, ax = plt.subplots(3, 1, sharex=True)
ax1, ax2, ax3 = ax[0], ax[1], ax[2]
ax1.plot(trounds[eqbTime+100:], SC_arr[0,eqbTime+100:],
         'bo-', label='SC')
ax1.plot(trounds[eqbTime+100:], UC_arr[0,eqbTime+100:],
         'go-', label='UC')
ax1.plot(trounds[eqbTime+100:], SD_arr[0,eqbTime+100:],
         'o-', label='SD', color='peru')
ax1.plot(trounds[eqbTime+100:], UD_arr[0,eqbTime+100:],
         'ro-', label='UD')
ax2.plot(trounds[eqbTime+100:], SDtUC_arr[0,0,eqbTime+100:],
         'o-', label=r'SD$\Rightarrow$UC', c='green')
ax2.plot(trounds[eqbTime+100:], UCtUD_arr[0,0,eqbTime+100:],
         'o-', label=r'UC$\Rightarrow$UD', c='peru')
ax2.plot(trounds[eqbTime+100:], UDtSD_arr[0,0,eqbTime+100:],
         'o-', label=r'UD$\Rightarrow$SD', c='red')
ax3.plot(trounds[eqbTime+100:], SCtUD_arr[0,0,eqbTime+100:],
         'o-', label=r'SC$\Rightarrow$UD', c='peru')
ax3.plot(trounds[eqbTime+100:], UDtUC_arr[0,0,eqbTime+100:],
         'o-', label=r'UD$\Rightarrow$UC', c='green')
ax3.plot(trounds[eqbTime+100:], UCtSC_arr[0,0,eqbTime+100:],
         'o-', label=r'UC$\Rightarrow$SC', c='blue')

ax3.set_xlabel('rounds', fontsize=15)
ax1.legend(loc=(-0.1,0.95), prop={'size': 15}, ncol=4, framealpha=0)
ax2.legend(loc=(-0.1,0.95), prop={'size': 15}, ncol=3, framealpha=0)
ax3.legend(loc=(-0.1,0.95), prop={'size': 15}, ncol=3, framealpha=0)
ax2.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.tick_params(axis='x', labelsize=15)
plt.subplots_adjust(left=0.1,
                    bottom=0.12, 
                    right=0.98, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.5)
plt.savefig("SatActTransitions_cyc1&2_HI_h"+str(hab)+"_re"+str(Re)+".tif")
plt.show()
