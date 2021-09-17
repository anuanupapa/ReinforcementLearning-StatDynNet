import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
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

sns.set()

tTime = time.time()
totTime = 0.

totNP = 500
trials = 5
rounds = 750

eps = 0.02
c = 1.
r = 2.
b = r*c
ini_re = 0.3

h_arr = np.arange(0.025, 0.99, 0.025)
h_arr = np.round(h_arr, 5)
beta_arr = np.logspace(-2.2, 0, 40)/c
print(h_arr, beta_arr)
beta_arr = beta_arr[::-1]

h1_arr = h_arr[::2]
b_arr = beta_arr[::2]
b1_arr=[]
for b in b_arr:
    b1_arr.append(np.format_float_scientific(b, trim='-'))

eqbTime = 250

print(len(beta_arr), len(h_arr))

coopfrac_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
sat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
asp_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Csat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Dsat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Casp_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Dasp_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))

hInd = -1
for hab in h_arr:
    hTime = time.time()
    print('start hab=', hab)
    hInd = hInd + 1
    betaInd = -1
    for bet in beta_arr:
        print('start beta=', bet)
        betaInd = betaInd + 1
        for it in range(trials):
            #print('trial number : '+str(it))
            AdjMat = init.init_adjmat(totNP, ini_re)
            [aspA, satA,
             pcA, payA,
             cpayA, habA,
             betaA, actA] = init.init_arr(totNP, AdjMat, 
                                          bet, hab, r, c)
            #actP = actA.copy()
            
            for i_main in range(rounds):
                
                pay = PGG.game(AdjMat, actA, r, c, totNP)
                
                [coopfrac, sais, asp, Csat, Dsat,
                 Casp, Dasp] = Record.measure(actA, satA, aspA, AdjMat)
                coopfrac_arr[betaInd, hInd, it, i_main] = coopfrac
                sat_arr[betaInd, hInd, it, i_main] = sais
                asp_arr[betaInd, hInd, it, i_main] = asp
                Csat_arr[betaInd, hInd, it, i_main] = Csat
                Dsat_arr[betaInd, hInd, it, i_main] = Dsat
                Casp_arr[betaInd, hInd, it, i_main] = Casp
                Dasp_arr[betaInd, hInd, it, i_main] = Dasp

                #actP = actA.copy()
                
                [aspA, satA,
                 pcA, actA,
                 cpayA] = Record.update_iterated(pay, aspA,
                                                 satA, payA,
                                                 cpayA, pcA,
                                                 habA, betaA,
                                                 actA, eps,
                                                 totNP)

    print('h='+str(hab)+' required: ', (time.time()-hTime))
    print('total time taken : ', (time.time()-tTime))
#coopfracW = windowing.window(coopfrac_arr, 100)
#satW = windowing.window(sat_arr, 100)
print(time.time()-tTime)
trounds = np.arange(0, rounds, 1)


# ---------------------------------------------
# Save numpy arrays in files
# ---------------------------------------------
#arrays = TemporaryFile()

np.savez("Fig1_BetaHabHeatmap.npz",beta=beta_arr, h=h_arr,
         coopfrac=coopfrac_arr, sat=sat_arr,
         Csat=Csat_arr, Dsat=Dsat_arr)
# --------------------------------------------

coopfracheat = np.mean(np.mean(coopfrac_arr[:,:,:,eqbTime:], axis=-1),
                       axis=-1)
xlabels = ['{:,.1f}'.format(x1) for x1 in h_arr[::1]]
ylabels = ['{:,.1e}'.format(y1) for y1 in beta_arr[::1]]
ax = sns.heatmap(coopfracheat, vmin=0, vmax=1, xticklabels=xlabels,
                 cmap='RdBu', yticklabels=ylabels)
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
plt.subplots_adjust(right=.98, left=0.2, bottom=0.13)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('sensitivity'+r'($\beta$)', fontsize=15)
ax.yaxis.set_label_coords(-0.25, 0.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Cooperator Fraction', fontsize=15)
#plt.savefig('Fig1a_coopfracheat_betahab.png')
plt.show()


'''
plt.clf()
satheat = np.mean(np.mean(sat_arr[:,:,:,eqbTime:], axis=-1),
                       axis=-1)
ax = sns.heatmap(satheat, vmin=-0.15, vmax=0.15, xticklabels=xlabels,
                 cbar_kws={'label': 'Average Satisfaction'},
                 cmap='PiYG', yticklabels=ylabels)
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
plt.subplots_adjust(right=.96, left=0.17)
plt.xlabel('habituation(h)')
plt.ylabel(r'$\beta$')
plt.savefig('satheat_betahab.png')
plt.show()
'''

Csatheat = np.mean(np.mean(Csat_arr[:,:,:,eqbTime:], axis=-1),
                   axis=-1)
Dsatheat = np.mean(np.mean(Dsat_arr[:,:,:,eqbTime:], axis=-1),
                   axis=-1)


Caspheat = np.mean(np.mean(Casp_arr[:,:,:,eqbTime:], axis=-1),
                   axis=-1)
Daspheat = np.mean(np.mean(Dasp_arr[:,:,:,eqbTime:], axis=-1),
                   axis=-1)


satlim = max([np.abs(np.min(Csatheat)), np.abs(np.max(Csatheat)),
               np.abs(np.min(Dsatheat)), np.abs(np.max(Dsatheat))])

ax = sns.heatmap(Csatheat, vmin=-1*satlim, vmax=satlim,
                 yticklabels=ylabels, xticklabels=xlabels, cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.96, left=0.2, bottom=0.13)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Average Satisfaction of Cooperators', fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('sensitivity'+r'($\beta$)', fontsize=15)
ax.yaxis.set_label_coords(-0.25, 0.5)
#plt.savefig('Fig1b_Csatheat_betahab.png')
plt.show()


ax = sns.heatmap(Dsatheat, vmin=-1*satlim, vmax=satlim,
                 xticklabels=xlabels, yticklabels=ylabels, cmap='PiYG')
ax.set_xticks(ax.get_xticks()[::4])
ax.set_yticks(ax.get_yticks()[::4])
ax.set_xticklabels(xlabels[::4], rotation='horizontal')
ax.set_yticklabels(ylabels[::4], rotation='horizontal')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.subplots_adjust(right=.96, left=0.2, bottom=0.13)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('habituation'+r'($h$)', fontsize=15)
plt.ylabel('sensitivity'+r'($\beta$)', fontsize=15)
plt.title('Average Satisfaction of Defectors', fontsize=15)
ax.yaxis.set_label_coords(-0.25, 0.5)
#plt.savefig('Fig1c_Dsatheat_betahab.png')
plt.show()
