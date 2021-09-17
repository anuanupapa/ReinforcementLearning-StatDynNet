import numpy as np
import numba as nb

@nb.njit
def feature(sat, act):
    cate = SatAct(sat, act)
    cate[np.where(cate==3.)[0]] = 4.
    cate[np.where(cate==-3.)[0]] = -2.

    SCpos = np.where(cate==4)[0]
    UCpos = np.where(cate==2)[0]
    SDpos = np.where(cate==-2)[0]
    UDpos = np.where(cate==-4)[0]
    SC = len(np.where(cate==4)[0])
    UC = len(np.where(cate==2)[0])
    SD = len(np.where(cate==-2)[0])
    UD = len(np.where(cate==-4)[0])
    return SC, UC, SD, UD, cate, SCpos, UCpos, SDpos, UDpos
    
@nb.vectorize
def SatAct(s, a):
# SC -> 1 + 3 = 4; SD -> 1 - 3 = -2
# UC -> -1 + 3 = 2; UD -> -1 -3 = -4
# SC -> 0 + 3 = 3; SD -> 0 - 3 = -3
    ssign = nb.float64(np.sign(s)) #Problematic when s=0.
    asign = nb.float64(np.sign(a-0.5))
    
    return(ssign + asign*3)
