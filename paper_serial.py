import numpy as np 
import ipdb 
import networkx as nx 
import threading
from collections import Counter 

def nbrs_for_v(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()


def peel(frontier, k, G, degrees): 
    f_next = []
    for v in frontier:
        for u in nbrs_for_v(G, v): 
            degrees[u] = degrees[u] - 1
            if degrees[u] == k: 
                f_next.append(u)
    return f_next, degrees

def k_core_decomposition_serial(G):
    # ipdb.set_trace() 
    # G = G - np.eye(G.shape[0]) 
    degrees = np.sum(G, axis=1) - 1
    active_set = list(range(G.shape[0]))
    coreness = np.zeros_like(degrees)
    
    k = 0 
    while list(active_set): 
        # frontier = np.intersect1d(np.argwhere(degrees == k), active_set)
        frontier = np.array([v for v in active_set if degrees[v] == k]) 
        # ipdb.set_trace()
        print(f"k = {k}")
        print(f'F: {frontier}')
        # print(active_set)
        print(f"D: {degrees}")
        while list(frontier): 
            for v in frontier: 
                coreness[v] = k
            frontier, degrees = peel(frontier, k, G, degrees)
        active_set = np.intersect1d(np.argwhere(degrees > k), active_set)
        
        k += 1         
    return coreness 

print("SERIAL")
# adj = np.load('/Users/rebeccasalganik/Documents/School/2025/Distributed/Data/Karate.npy').astype(int)
adj = np.array([
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
degree = np.sum(adj, axis=1)
core_numbers = k_core_decomposition_serial(adj) + 1
print(core_numbers)


# adj = np.load('/Users/rebeccasalganik/Documents/School/2025/Distributed/Data/Karate.npy')
adj = np.array([
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
H = nx.from_numpy_array(adj) #nx.havel_hakimi_graph(degrees)
solution = list(nx.core_number(H).values())

print((core_numbers == solution).all()) 