import numpy as np
import numba as nb
from numba import float64
from numba import guvectorize


@nb.njit
def measure(action, sat, asp, AM):

    Cplayers = np.where(action == 1.)[0]
    Dplayers = np.where(action == 0.)[0]
    Casp = np.sum(asp[Cplayers])/(len(Cplayers)+0.01)
    Dasp = np.sum(asp[Dplayers])/(len(Dplayers)+0.01)
    Csat = np.sum(sat[Cplayers])/(len(Cplayers)+0.01)
    Dsat = np.sum(sat[Dplayers])/(len(Dplayers)+0.01)
    aspiration = np.sum(asp)/len(action)
    satisfaction = np.sum(sat)/len(action)

    return(np.sum(action)/len(action), satisfaction, aspiration,
           Csat, Dsat, Casp, Dasp)


@nb.njit
def update_iterated(payoff, asp, sat, pay, cpay, pC, hab, beta, act, tremble, N):

    aspM = np.zeros_like(asp)
    satM = np.zeros_like(sat)
    pCM = np.zeros_like(pC)
    actM = np.zeros_like(act)
    cpayM = np.zeros_like(cpay)

    cpayM = update_cpay(cpay, payoff)
    satM = update_sat(beta, payoff, asp)
    pCM = update_pC(act, satM, pC)
    actM = update_act(pCM, tremble) # this is becoming bool
    aspM = update_asp(hab, asp, payoff)

    return aspM, satM, pCM, actM, cpayM
    
@nb.vectorize
def update_pay(p):
    payMod = p
    return payMod

@nb.vectorize
def update_cpay(cp, p):
    cpayMod = cp + p
    return cpayMod

@nb.vectorize
def update_sat(b, p, a):
    satMod = np.tanh(b * (p - a))
    return satMod

@nb.vectorize
def update_pC(a, s, p):
    if s >= 0. and a >= 0.5:
        pM = p + (1 - p) * s
    elif s < 0. and a >= 0.5:
        pM = p + s * p
    elif s >= 0. and a < 0.5:
        pM = p - s * p
    else:
        pM = p - (1. - p) * s
    return pM

@nb.vectorize
def update_act(p, t):
    if np.random.random() > t:
        aM = nb.float64(np.random.random() <= p)
    else:
        aM = nb.float64(np.random.random() > p)
    return aM

@nb.vectorize
def update_asp(h, a, p):
    aM = (1 - h) * a + h * p
    return aM


@nb.njit
def check(arr):
    array = np.zeros_like(arr)
    array = vect(arr)
    return(array)

@nb.vectorize
def vect(x):
    x=5
    return(x)
    
if __name__ == "__main__":
    act = np.ones(5)
    sats = np.ones(5)
    act[3]=0.
    sats[2]=0.
    am = 0.
    mea = measure(act, sats, am)
    print(mea)
    a = np.array([1,2,3,4])
    b = np.array([100,200,300,400])
    #print(update(a, b))
    print(check(a))
