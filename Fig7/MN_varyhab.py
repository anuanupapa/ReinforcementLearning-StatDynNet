import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as scsp
import networkx as nx
import time
from datetime import datetime
import numba as nb
from numba import int64, float64
from tempfile import TemporaryFile
import seaborn as sns
import PGG
import Record_vectorized as Record
import rewire
import initialize as init
import features
import eval_shift


sns.set(style="whitegrid")

tTime = time.time()
totTime = 0.

totNP = 500
trials = 25
rounds = 250

eps = 0.02
c = 1.
r = 2.
b = r*c
ini_re = 0.3
re = 0.95
h_arr = np.arange(0.0, 1.0001, 0.025)
h_arr = np.round(h_arr, 5)
bet = 10**(0.)/c
eqbTime = 200
var_arr = h_arr
print(h_arr)

coopfrac_arr = np.zeros((len(var_arr), trials, rounds))
sat_arr = np.zeros((len(var_arr), trials, rounds))
SC_arr = np.zeros((len(var_arr), trials, rounds))
UC_arr = np.zeros((len(var_arr), trials, rounds))
SD_arr = np.zeros((len(var_arr), trials, rounds))
UD_arr = np.zeros((len(var_arr), trials, rounds))
category_arr = np.zeros((len(var_arr), trials, rounds, totNP, 1))
Csat_arr = np.zeros((len(var_arr), trials, rounds))
Dsat_arr = np.zeros((len(var_arr), trials, rounds))

MISS=0.
varInd = -1
for hab in h_arr:
    
    hTime = time.time()
    print('hab: ', hab)
    varInd = varInd + 1

    for it in range(trials):
        print('trial no:', it)
        AdjMat = init.init_adjmat(totNP, ini_re)
        [aspA, satA,
         pcA, payA,
         cpayA, habA,
         betaA, actA] = init.init_arr(totNP, AdjMat, 
                                      bet, hab, r, c)

        for i_main in range(rounds):

            pay = PGG.game(AdjMat, actA, r, c, totNP)

            [coopfrac,
             sais] = Record.measure(actA, satA, AdjMat)
            coopfrac_arr[varInd, it, i_main] = coopfrac
            sat_arr[varInd, it, i_main] = sais
            Csat = np.mean(satA[np.where(actA==1)[0]])
            Csat_arr[varInd, it, i_main] = Csat
            Dsat = np.mean(satA[np.where(actA==0)[0]])
            Dsat_arr[varInd, it, i_main] = Dsat

            AdjMat = rewire.rewiring_process(AdjMat, actA, re)
            
            [aspA, satA,
             pcA, actA,
             cpayA] = Record.update_iterated(pay, aspA,
                                             satA, payA,
                                             cpayA, pcA,
                                             habA, betaA,
                                             actA, eps,
                                             totNP)
            
            [SC,UC,SD,UD,
             category, missed] = features.feature(satA, actA)
            category_arr[varInd, it, i_main, :] = category
            SC_arr[varInd, it, i_main] = SC
            UC_arr[varInd, it, i_main] = UC
            SD_arr[varInd, it, i_main] = SD
            UD_arr[varInd, it, i_main] = UD
            MISS = MISS + missed
            
    print('hab='+str(hab)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))

print(MISS)
#coopfracW = windowing.window(coopfrac_arr, 100)
#satW = windowing.window(sat_arr, 100)

# Category arrays ------------------
coopfrac_trials = np.mean(coopfrac_arr[:,:,eqbTime:], axis=-1)
sat_trials = np.mean(sat_arr[:,:,eqbTime:], axis=-1)
Csat_trials = np.mean(Csat_arr[:,:,eqbTime:], axis=-1)
Dsat_trials = np.mean(Dsat_arr[:,:,eqbTime:], axis=-1)
cfmean_arr = np.mean(coopfrac_trials, axis=-1)
satmean_arr = np.mean(sat_trials, axis=-1)
Csatmean_arr = np.mean(Csat_trials, axis=-1)
Dsatmean_arr = np.mean(Dsat_trials, axis=-1)
cfstd_arr = np.std(coopfrac_trials, axis=-1)
satstd_arr = np.std(sat_trials, axis=-1)
Csatstd_arr = np.std(Csat_trials, axis=-1)
Dsatstd_arr = np.std(Dsat_trials, axis=-1)

SC_trials = np.mean(SC_arr[:,:,eqbTime:], axis=-1)
UC_trials = np.mean(UC_arr[:,:,eqbTime:], axis=-1)
SD_trials = np.mean(SD_arr[:,:,eqbTime:], axis=-1)
UD_trials = np.mean(UD_arr[:,:,eqbTime:], axis=-1)
SCmean_arr = np.mean(SC_trials, axis=-1)/totNP
UCmean_arr = np.mean(UC_trials, axis=-1)/totNP
SDmean_arr = np.mean(SD_trials, axis=-1)/totNP
UDmean_arr = np.mean(UD_trials, axis=-1)/totNP
SCstd_arr = np.std(SC_trials, axis=-1)/totNP
UCstd_arr = np.std(UC_trials, axis=-1)/totNP
SDstd_arr = np.std(SD_trials, axis=-1)/totNP
UDstd_arr = np.std(UD_trials, axis=-1)/totNP
# ------------------------------------


# ---------------------------------------------------------
# Player class shifts
# ---------------------------------------------------------
[SCtSD_arr, SCtUC_arr, SCtUD_arr,
 SDtSC_arr, SDtUC_arr, SDtUD_arr,
 UCtSC_arr, UCtSD_arr, UCtUD_arr,
 UDtSC_arr, UDtSD_arr, UDtUC_arr,
 SCtSC_arr, SDtSD_arr,
 UCtUC_arr, UDtUD_arr] = eval_shift.ClassShift(category_arr)

np.savez("rawdata_PT1_arrays_hab_beta"
         +str(bet)+"_re"+str(re)+".npz",h=h_arr,
         cfrac=coopfrac_arr, Csat = Csat_arr, Dsat = Dsat_arr,

         SC = SC_arr, UC = UC_arr, SD = SD_arr, UD = UD_arr,
         
         SCtSC=SCtSC_arr, SCtSD=SCtSD_arr,
         SCtUC=SCtUC_arr, SCtUD=SCtUD_arr,
         
         SDtSC=SDtSC_arr, SDtSD=SDtSD_arr,
         SDtUC=SDtUC_arr, SDtUD=SDtUD_arr,
         
         UCtSC=UCtSC_arr, UCtSD=UCtSD_arr,
         UCtUC=UCtUC_arr, UCtUD=UCtUD_arr,
         
         UDtSC=UDtSC_arr, UDtSD=UDtSD_arr,
         UDtUC=UDtUC_arr, UDtUD=UDtUD_arr)

SCtSD_trials = np.mean(SCtSD_arr[:,:,eqbTime:], axis=-1)
SCtSDmean_arr = np.mean(SCtSD_trials, axis=-1)
SCtSDstd_arr = np.std(SCtSD_trials, axis=-1)
SCtUC_trials = np.mean(SCtUC_arr[:,:,eqbTime:], axis=-1)
SCtUCmean_arr = np.mean(SCtUC_trials, axis=-1)
SCtUCstd_arr = np.std(SCtUC_trials, axis=-1)
SCtUD_trials = np.mean(SCtUD_arr[:,:,eqbTime:], axis=-1)
SCtUDmean_arr = np.mean(SCtUD_trials, axis=-1)
SCtUDstd_arr = np.std(SCtUD_trials, axis=-1)
SCtSC_trials = np.mean(SCtSC_arr[:,:,eqbTime:], axis=-1)
SCtSCmean_arr = np.mean(SCtSC_trials, axis=-1)
SCtSCstd_arr = np.std(SCtSC_trials, axis=-1)

SDtSC_trials = np.mean(SDtSC_arr[:,:,eqbTime:], axis=-1)
SDtSCmean_arr = np.mean(SDtSC_trials, axis=-1)
SDtSCstd_arr = np.std(SDtSC_trials, axis=-1)
SDtUC_trials = np.mean(SDtUC_arr[:,:,eqbTime:], axis=-1)
SDtUCmean_arr = np.mean(SDtUC_trials, axis=-1)
SDtUCstd_arr = np.std(SDtUC_trials, axis=-1)
SDtUD_trials = np.mean(SDtUD_arr[:,:,eqbTime:], axis=-1)
SDtUDmean_arr = np.mean(SDtUD_trials, axis=-1)
SDtUDstd_arr = np.std(SDtUD_trials, axis=-1)
SDtSD_trials = np.mean(SDtSD_arr[:,:,eqbTime:], axis=-1)
SDtSDmean_arr = np.mean(SDtSD_trials, axis=-1)
SDtSDstd_arr = np.std(SDtSD_trials, axis=-1)

UCtSC_trials = np.mean(UCtSC_arr[:,:,eqbTime:], axis=-1)
UCtSCmean_arr = np.mean(UCtSC_trials, axis=-1)
UCtSCstd_arr = np.std(UCtSC_trials, axis=-1)
UCtSD_trials = np.mean(UCtSD_arr[:,:,eqbTime:], axis=-1)
UCtSDmean_arr = np.mean(UCtSD_trials, axis=-1)
UCtSDstd_arr = np.std(UCtSD_trials, axis=-1)
UCtUD_trials = np.mean(UCtUD_arr[:,:,eqbTime:], axis=-1)
UCtUDmean_arr = np.mean(UCtUD_trials, axis=-1)
UCtUDstd_arr = np.std(UCtUD_trials, axis=-1)
UCtUC_trials = np.mean(UCtUC_arr[:,:,eqbTime:], axis=-1)
UCtUCmean_arr = np.mean(UCtUC_trials, axis=-1)
UCtUCstd_arr = np.std(UCtUC_trials, axis=-1)

UDtSC_trials = np.mean(UDtSC_arr[:,:,eqbTime:], axis=-1)
UDtSCmean_arr = np.mean(UDtSC_trials, axis=-1)
UDtSCstd_arr = np.std(UDtSC_trials, axis=-1)
UDtSD_trials = np.mean(UDtSD_arr[:,:,eqbTime:], axis=-1)
UDtSDmean_arr = np.mean(UDtSD_trials, axis=-1)
UDtSDstd_arr = np.std(UDtSD_trials, axis=-1)
UDtUC_trials = np.mean(UDtUC_arr[:,:,eqbTime:], axis=-1)
UDtUCmean_arr = np.mean(UDtUC_trials, axis=-1)
UDtUCstd_arr = np.std(UDtUC_trials, axis=-1)
UDtUD_trials = np.mean(UDtUD_arr[:,:,eqbTime:], axis=-1)
UDtUDmean_arr = np.mean(UDtUD_trials, axis=-1)
UDtUDstd_arr = np.std(UDtUD_trials, axis=-1)


# ---------------------------------------------
# Save numpy arrays in files
# ---------------------------------------------
#arrays = TemporaryFile()

np.savez("PT1_arrays_hab_beta"
         +str(bet)+"_re"+str(re)+".npz",h=h_arr,

         coopfracmean = cfmean_arr, coopfracstd =cfstd_arr,

         Csatmean = Csatmean_arr, Dsatmean = Dsatmean_arr,
         Csatstd = Csatstd_arr, Dsatstd = Dsatstd_arr,
         
         SCmean=SCmean_arr, SDmean=SDmean_arr,
         UCmean=UCmean_arr, UDmean=UDmean_arr,
         SCstd=SCstd_arr, SDstd=SDstd_arr,
         UCstd=UCstd_arr, UDstd=UDstd_arr,
         
         SCtSCmean=SCtSCmean_arr, SCtSDmean=SCtSDmean_arr,
         SCtUCmean=SCtUCmean_arr, SCtUDmean=SCtUDmean_arr,
         SCtSCstd=SCtSCstd_arr, SCtSDstd=SCtSDstd_arr,
         SCtUCstd=SCtUCstd_arr, SCtUDstd=SCtUDstd_arr,
         
         SDtSCmean=SDtSCmean_arr, SDtSDmean=SDtSDmean_arr,
         SDtUCmean=SDtUCmean_arr, SDtUDmean=SDtUDmean_arr,
         SDtSCstd=SDtSCstd_arr, SDtSDstd=SDtSDstd_arr,
         SDtUCstd=SDtUCstd_arr, SDtUDstd=SDtUDstd_arr,
         
         UCtSCmean=UCtSCmean_arr, UCtSDmean=UCtSDmean_arr,
         UCtUCmean=UCtUCmean_arr, UCtUDmean=UCtUDmean_arr,
         UCtSCstd=UCtSCstd_arr, UCtSDstd=UCtSDstd_arr,
         UCtUCstd=UCtUCstd_arr, UCtUDstd=UCtUDstd_arr,
         
         UDtSCmean=UDtSCmean_arr, UDtSDmean=UDtSDmean_arr,
         UDtUCmean=UDtUCmean_arr, UDtUDmean=UDtUDmean_arr,
         UDtSCstd=UDtSCstd_arr, UDtSDstd=UDtSDstd_arr,
         UDtUCstd=UDtUCstd_arr, UDtUDstd=UDtUDstd_arr
         )
# --------------------------------------------------

'''
# FIXED STATE TO FOUR STATES ---------------
# SC TO OTHER 

plt.plot(var_arr, SCtSCmean_arr, 'o-',
         label = r'SC$\Rightarrow$SC', color='b')
plt.fill_between(var_arr, SCtSCmean_arr-SCtSCstd_arr,
                 SCtSCmean_arr+SCtSCstd_arr, alpha=0.3, color="b")

plt.plot(var_arr, SCtUDmean_arr, 'o-',
         label = r'SC$\Rightarrow$UD', color='g')
plt.fill_between(var_arr, SCtUDmean_arr-SCtUDstd_arr,
                 SCtUDmean_arr+SCtUDstd_arr, alpha=0.3, color="g")

plt.plot(var_arr, SCtUCmean_arr, 'o-', label =
         r'SC$\Rightarrow$UC', color="peru")
plt.fill_between(var_arr, SCtUCmean_arr-SCtUCstd_arr,
                 SCtUCmean_arr+SCtUCstd_arr, alpha=0.3, color="peru")

plt.plot(var_arr, SCtSDmean_arr, 'o-',
         label = r'SC$\Rightarrow$SD', color='r')
plt.fill_between(var_arr, SCtSDmean_arr-SCtSDstd_arr,
                 SCtSDmean_arr+SCtSDstd_arr, alpha=0.3, color="r")

plt.xlabel('habituation(h)')
plt.ylabel('Number of Transitions from SC')
plt.legend()
plt.ylim(0,140)
#plt.savefig('PT1_SCtoOthersVShab_beta'+str(bet)+'_re'+str(re)+'.png')
#plt.show()

# SD TO OTHERS
plt.plot(var_arr, SDtSCmean_arr, 'o-',
         label = r'SD$\Rightarrow$SC', color='b')
plt.fill_between(var_arr, SDtSCmean_arr-SDtSCstd_arr,
                 SDtSCmean_arr+SDtSCstd_arr, alpha=0.3, color="b")

plt.plot(var_arr, SDtUDmean_arr, 'go-', label = r'SD$\Rightarrow$UD')
plt.fill_between(var_arr, SDtUDmean_arr-SDtUDstd_arr,
                 SDtUDmean_arr+SDtUDstd_arr, alpha=0.3, color="g")

plt.plot(var_arr, SDtUCmean_arr, 'o-',
         label = r'SD$\Rightarrow$UC', color='peru')
plt.fill_between(var_arr, SDtUCmean_arr-SDtUCstd_arr,
                 SDtUCmean_arr+SDtUCstd_arr, alpha=0.3, color="peru")

plt.plot(var_arr, SDtSDmean_arr, 'ro-', label = r'SD$\Rightarrow$SD')
plt.fill_between(var_arr, SDtSDmean_arr-SDtSDstd_arr,
                 SDtSDmean_arr+SDtSDstd_arr, alpha=0.3, color="r")

plt.xlabel('habituation(h)')
plt.ylabel('Number of Transitions from SD')
plt.legend()
plt.ylim(0,140)
#plt.savefig('PT1_SDtoOthersVShab_beta'+str(bet)+'_re'+str(re)+'.png')
#plt.show()

#UC TO OTHERS
plt.plot(var_arr, UCtSCmean_arr, 'bo-', label = r'UC$\Rightarrow$SC')
plt.fill_between(var_arr, UCtSCmean_arr-UCtSCstd_arr,
                 UCtSCmean_arr+UCtSCstd_arr, alpha=0.3, color="b")

plt.plot(var_arr, UCtUDmean_arr, 'go-', label = r'UC$\Rightarrow$UD')
plt.fill_between(var_arr, UCtUDmean_arr-UCtUDstd_arr,
                 UCtUDmean_arr+UCtUDstd_arr, alpha=0.3, color="g")

plt.plot(var_arr, UCtUCmean_arr, 'o-',
         label = r'UC$\Rightarrow$UC', color="peru")
plt.fill_between(var_arr, UCtUCmean_arr-UCtUCstd_arr,
                 UCtUCmean_arr+UCtUCstd_arr, alpha=0.3, color="peru")

plt.plot(var_arr, UCtSDmean_arr, 'ro-', label = r'UC$\Rightarrow$SD')
plt.fill_between(var_arr, UCtSDmean_arr-UCtSDstd_arr,
                 UCtSDmean_arr+UCtSDstd_arr, alpha=0.3, color="r")

plt.xlabel('habituation(h)')
plt.ylabel('Number of Transitions from UC')
plt.legend()
plt.ylim(0,140)
#plt.savefig('PT1_UCtoOthersVShab_beta'+str(bet)+'_re'+str(re)+'.png')
#plt.show()

#UD TO OTHERS 
plt.plot(var_arr, UDtSCmean_arr, 'bo-', label = r'UD$\Rightarrow$SC')
plt.fill_between(var_arr, UDtSCmean_arr-UDtSCstd_arr,
                 UDtSCmean_arr+UDtSCstd_arr, alpha=0.3, color="b")

plt.plot(var_arr, UDtUDmean_arr, 'go-', label = r'UD$\Rightarrow$UD')
plt.fill_between(var_arr, UDtUDmean_arr-UDtUDstd_arr,
                 UDtUDmean_arr+UDtUDstd_arr, alpha=0.3, color="g")

plt.plot(var_arr, UDtUCmean_arr, 'o-',
         label = r'UD$\Rightarrow$UC', color="peru")
plt.fill_between(var_arr, UDtUCmean_arr-UDtUCstd_arr,
                 UDtUCmean_arr+UDtUCstd_arr, alpha=0.3, color="peru")

plt.plot(var_arr, UDtSDmean_arr, 'ro-', label = r'UD$\Rightarrow$SD')
plt.fill_between(var_arr, UDtSDmean_arr-UDtSDstd_arr,
                 UDtSDmean_arr+UDtSDstd_arr, alpha=0.3, color="r")


plt.xlabel('habituation(h)')
plt.ylabel('Number of transitions from UD')
plt.legend()
#plt.savefig('PT1_UDtoOthersVShab_beta'+str(bet)+'_re'+str(re)+'.png')
#plt.show()
# --------------------------------------------------------



# Plotting ---------------------------
plt.plot(var_arr, SCmean_arr, 'bo-', label = 'SC')
plt.fill_between(var_arr, SCmean_arr-SCstd_arr,
                 SCmean_arr+SCstd_arr, alpha=0.3, color="b")

plt.plot(var_arr, UDmean_arr, 'go-', label = 'UD')
plt.fill_between(var_arr, UDmean_arr-UDstd_arr,
                 UDmean_arr+UDstd_arr, alpha=0.3, color="g")

plt.plot(var_arr, UCmean_arr, 'o-', label = 'UC', color="peru")
plt.fill_between(var_arr, UCmean_arr-UCstd_arr,
                 UCmean_arr+UCstd_arr, alpha=0.3, color="peru")

plt.plot(var_arr, SDmean_arr, 'o-', label = 'SD', color="r")
plt.fill_between(var_arr, SDmean_arr-SDstd_arr,
                 SDmean_arr+SDstd_arr, alpha=0.3, color="r")

plt.xlabel('habituation(h)')
plt.ylabel('Fraction of Player in Category')
plt.legend(loc=(0, 0.5))
#plt.savefig('PT1_satactVShab_beta'+str(bet)+'_re'+str(re)+'.png')
#plt.show()


plt.plot(var_arr, cfmean_arr, 'bo-')
plt.fill_between(var_arr, cfmean_arr-cfstd_arr,
                 cfmean_arr+cfstd_arr, alpha=0.3, color="b")
plt.title(r'$\beta=$'+str(bet)+' N='+str(totNP))
plt.xlabel('habituation')
plt.ylabel('Average Fraction of cooperators')
#plt.savefig('coopfracVShab_beta'+str(bet)+'.png')
plt.show()

plt.plot(var_arr, satmean_arr, 'go-')
plt.fill_between(var_arr, satmean_arr-satstd_arr,
                 satmean_arr+satstd_arr, alpha=0.3, color="g")
plt.title(r'$\beta=$'+str(bet)+' N='+str(totNP))
plt.xlabel('habituation')
plt.ylabel("Average Satisfaction")
#plt.savefig('satVShab_beta'+str(bet)+'.png')
#plt.show()


plt.plot(var_arr, Csatmean_arr, 'o-', c='limegreen', label='Average Satisfaction of Cooperators')
plt.fill_between(var_arr, Csatmean_arr-Csatstd_arr,
                 Csatmean_arr+Csatstd_arr, alpha=0.3,
                 color="limegreen")
plt.title(r'$\beta=$'+str(bet)+' N='+str(totNP))
plt.xlabel('habituation')
#plt.ylabel('Average Satisfaction of Cooperators')
#plt.savefig('CsatVShab_beta'+str(bet)+'.png')
#plt.show()

plt.plot(var_arr, Dsatmean_arr, 'o-', c='darkgreen', label='Average Satisfaction of Defectors')
plt.fill_between(var_arr, Dsatmean_arr-Dsatstd_arr,
                 Dsatmean_arr+Dsatstd_arr, alpha=0.3,
                 color="darkgreen")
plt.title(r'$\beta=$'+str(bet)+' N='+str(totNP))
plt.xlabel('habituation')
#plt.ylabel("Average Satisfaction of Defectors")
#plt.ylim(-0.35,0.35)
plt.legend()
#plt.savefig('CDsatVShab_beta'+str(bet)+'.png')
#plt.show()
plt.clf()

# -----------------------------------------------------------


'''
