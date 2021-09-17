
import numpy as np

import numba
#from numba import jitclass, int64
#-------------------------------------------------
#game : 1 round of PGG over the entire network for all nodes
#-------------------------------------------------
@numba.njit
def game(AM, pRes, r, m, N):#pRes:Player Response AM:Adjacency Matrix
    sumarr = np.zeros_like(pRes)
    player_count = np.zeros_like(pRes)
    #network on which the PGG will be played
    net_each_pgg=AM
    sumarr = np.sum(net_each_pgg, axis=1)#.reshape(N,1)

    #number of players participating in 'i'th PGG is player_count
    player_count=sumarr.reshape(N,1)

    pres=pRes.reshape(N,1)
    
    #PGG: Total PG for the game around 'i'th node is stored.
    #C=1 D=0, thus dot product works.
    PG_accumulated=r*m*net_each_pgg.dot(pres)
    PG_obt=PG_accumulated-m*np.multiply(player_count,pres)

    return(PG_obt)
#-----------------------------------------------------


if __name__=='__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.sparse import lil_matrix
    from numba import float64
    import scipy.sparse as scsp
    
    '''
    #1st config
    p1=player(action=1)
    p2=player(action=0)
    p3=player(action=1)
    p4=player(action=0)
    player_arr=np.array([p1,p2,p3,p4])
    adj_mat=np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
    '''
    
    #2nd config
    N=6
    p0=0.7
    player_arr=[]

    for q in range(N):
        player_arr.append(float(np.random.random()<=p0))
    player_arr = np.array(player_arr)
    
    G = nx.erdos_renyi_graph(N, 0.5)
    adj_mat=nx.adjacency_matrix(G)
    adj_mat = scsp.csr_matrix.toarray(adj_mat)
    adj_mat = adj_mat.astype('float64')
    
    payoff=game(adj_mat,player_arr, 2, 50, N)

    print(payoff)
    
    G=nx.from_numpy_matrix(adj_mat)
    col_map=[]
    for p in player_arr:
        if p==1:
            col_map.append('blue')
        else:
            col_map.append('red')
    nx.draw(G, node_color=col_map, with_labels=True, pos=nx.fruchterman_reingold_layout(G))
    plt.show()
