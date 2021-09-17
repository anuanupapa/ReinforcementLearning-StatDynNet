import numpy as np
import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
#matplotlib.use("Agg")
import networkx as nx


data = np.load("Fig2_PT1_varyhab.npz")

index = np.where(data["h"]==0.9)[0][0]
hab = data["h"][index]
print(index)

fig, ax = plt.subplots(figsize=(6,4))

G = nx.DiGraph()

plt.clf()
node_sizes = []
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
node_sizes.append(data['SCmean'][index]*1000)
node_sizes.append(data['SDmean'][index]*1000)
node_sizes.append(data['UCmean'][index]*1000)
node_sizes.append(data['UDmean'][index]*1000)

pos = {0: np.array([0, .5]), 1: np.array([0, -0.5]),
       2: np.array([.25, 0]), 3: np.array([-.25, 0])}

nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=["blue","red",
                                   "green", "peru"])

edges = []
for p1 in range(4):
    for p2 in range(p1+1,4):
        G.add_edge(p1, p2)
        G.add_edge(p2, p1)

edge_alphas = []
#edge_alphas.append(SCtSC[frame])
edge_alphas.append(data['SCtSDmean'][index])
edge_alphas.append(data['SCtUCmean'][index])
edge_alphas.append(data['SCtUDmean'][index])

edge_alphas.append(data['SDtSCmean'][index])
edge_alphas.append(data['SDtUCmean'][index])
edge_alphas.append(data['SDtUDmean'][index])

edge_alphas.append(data['UCtSCmean'][index])
edge_alphas.append(data['UCtSDmean'][index])
edge_alphas.append(data['UCtUDmean'][index])

edge_alphas.append(data['UDtSCmean'][index])
edge_alphas.append(data['UDtSDmean'][index])
edge_alphas.append(data['UDtUCmean'][index])

labels = {0: "SC", 1: "SD", 2: "UC", 3: "UD"}
nx.draw_networkx_labels(G, pos, labels, font_size=20)
edges = nx.draw_networkx_edges(G,pos,arrowstyle="-|>",arrowsize=50,
                               edge_color=edge_alphas,
                               edge_cmap=plt.cm.Greys,width=2,
                               connectionstyle='arc3, rad=0.1')

pc = mpl.collections.PatchCollection(
    edges,cmap=plt.cm.Greys)
pc.set_array(edge_alphas)
plt.colorbar(pc,
             label='Average number of category transitions per round')
plt.title(r'$\beta=$'+str(1)+
          r' $h$='+str(data['h'][index]), fontsize=20)
plt.subplots_adjust(right=1.02, left=0.02, bottom=0.02, top=0.92)
plt.savefig("Flowchart_h"+str(data["h"][index])+".png", dpi=300)
plt.show()

    
