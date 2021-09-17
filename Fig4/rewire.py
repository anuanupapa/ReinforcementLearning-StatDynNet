import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit
def rewiring_process(AdjMat, pRes, re, pr=0.87, pb=0.7, pm=0.93, pe=0.3, ps=0.2):
  #G=nx.convert_matrix.from_numpy_array(AdjMat)
  N=len(pRes)
  adjmat=AdjMat.copy()
  for ind1 in range(N-1): #The last node has all pairings done
    for ind2 in np.arange(ind1+1,N,1):
      decision_maker_selector=np.array([ind1,ind2])
      if np.random.random()<=re:
        if adjmat[ind1,ind2]==1:
          #print(decision_maker_selector)
          selInd=(np.random.choice(decision_maker_selector))
          if pRes[selInd]==1:
            rewRes=int(np.random.random()<=pr) #P_retain
            AdjMat[ind1, ind2]=rewRes
            AdjMat[ind2, ind1]=rewRes
          else:
            rewRes=int(np.random.random()>pb) #P_break
            AdjMat[ind1, ind2]=rewRes
            AdjMat[ind2, ind1]=rewRes
        else:
          if pRes[ind1]==1 and pRes[ind2]==1:
            rewRes=int(np.random.random()<=pm)
            AdjMat[ind1, ind2]=rewRes
            AdjMat[ind2, ind1]=rewRes
          elif pRes[ind1]*pRes[ind2]==-1:
            rewRes=int(np.random.random()<=pe)
            AdjMat[ind1, ind2]=rewRes
            AdjMat[ind2, ind1]=rewRes
          else:
            rewRes=int(np.random.random()<=ps)
            AdjMat[ind1, ind2]=rewRes
            AdjMat[ind2, ind1]=rewRes
      else:
        pass
    
  return(AdjMat)
