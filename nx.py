import networkx as nx 
degrees = [0, 1, 2, 2, 2, 2, 3]
H = nx.havel_hakimi_graph(degrees)
print(nx.k_core(H).nodes) 
