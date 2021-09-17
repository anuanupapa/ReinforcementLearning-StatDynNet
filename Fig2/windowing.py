import numpy as np
import numba as nb

def window(arr, window):
    N = np.shape(arr)[-1]
    windarr = np.zeros_like(arr)
    Ind = 0
    beginInd = Ind - np.int64(window/2)
    endInd = Ind + np.int64(window/2)
    
    while beginInd < 0:
        windarr[Ind] = np.mean(
            arr[0:endInd], axis=-1)
        endInd = endInd + 1
        beginInd = beginInd + 1
        Ind = Ind + 1

    while endInd < N:
        windarr[Ind] = np.mean(
            arr[beginInd:endInd], axis=-1)
        endInd = endInd + 1
        beginInd = beginInd + 1
        Ind = Ind + 1

    while Ind < N :
        windarr[Ind] = np.mean(
            arr[beginInd:N-1], axis=-1)
        endInd = endInd + 1
        beginInd = beginInd + 1
        Ind = Ind + 1
        
    return(windarr)
