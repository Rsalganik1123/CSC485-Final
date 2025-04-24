import networkx as nx 
import numpy as np 
import ipdb 

degrees = [0, 1, 2, 2, 2, 2, 3]
adj = np.load('/Users/rebeccasalganik/Documents/School/2025/Distributed/Data/Karate.npy')
H = nx.from_numpy_array(adj) #nx.havel_hakimi_graph(degrees)
print(nx.k_core(H).nodes) 
