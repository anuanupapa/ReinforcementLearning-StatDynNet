import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit
def rewiring_process(AdjMat, pSat, re, probs):
  #G=nx.convert_matrix.from_numpy_array(AdjMat)
  N=len(pSat)
  adjmat=AdjMat.copy()

  p1 = probs[0]
  p0 = probs[1]
  
  for ind1 in range(N-1): #The last node has all pairings done
    for ind2 in np.arange(ind1+1,N,1):

      if np.random.random()<=re:
        if pSat[ind1]>=0 and pSat[ind2]>=0:
          rewRes=int(np.random.random()<=p1*p1)
          AdjMat[ind1, ind2]=rewRes
          AdjMat[ind2, ind1]=rewRes
        elif pSat[ind1]*pSat[ind2]<0:
          rewRes=int(np.random.random()<=p1*p0)
          AdjMat[ind1, ind2]=rewRes
          AdjMat[ind2, ind1]=rewRes
        elif pSat[ind1]<=0 and pSat[ind2]<=0:
          rewRes=int(np.random.random()<=p0*p0)
          AdjMat[ind1, ind2]=rewRes
          AdjMat[ind2, ind1]=rewRes
        else:
          print(pSat[ind1], pSat[ind2])
          print("Neg")            
      else:
        pass
    
  return(AdjMat)
