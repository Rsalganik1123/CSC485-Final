import numpy as np 
import ipdb 
import threading
from collections import Counter 
import networkx as nx 

def nbrs_all_gather(G, frontier): 
    return np.argwhere(G[frontier[0], :] == 1).flatten().tolist()


def offline_peel_decr(H, idx, degrees, processed, k): 
    for u in idx:
        f_u = H[u]
        if not processed[u] and degrees[u] > k:
            degrees[u] = degrees[u] - f_u

def offline_peel(frontier, k, G, degrees, processed, num_threads=1):
    # Step 1: Gather all neighbors (with duplicates)
    # ipdb.set_trace() 
    L = nbrs_all_gather(G, frontier)

    # Step 2: Count frequency of each neighbor
    H = Counter(L)

    # Step 3: Decrease degrees of neighbors (if not already processed)
    mt = num_threads > 1 
    if len(H.keys()) > num_threads: 
        if mt: 
            thread_list = [] 
            data_chunk = np.array_split(list(H.keys()), num_threads-1)
            for i,t in enumerate(data_chunk):
                m = threading.Thread(target=offline_peel_decr, args=(H, t, degrees, processed, k)) 
                thread_list.append(m)
            for m in thread_list:
                m.start() 
            for m in thread_list:
                m.join()
        else: 
            offline_peel_decr(H, H.keys(), degrees, processed, k)
    else: 
        offline_peel_decr(H, H.keys(), degrees, processed, k)

    # Step 4: Next frontier: unprocessed nodes whose degree <= k
    f_next = np.array([u for u in set(L) if not processed[u] and degrees[u] <= k]) 
    return f_next, degrees

def k_core_decomposition_julienne(G, peel_fnc, num_threads=1):
    degrees = np.sum(G, axis=1) 
    coreness = np.zeros_like(degrees)
    processed = np.zeros_like(degrees, dtype=bool)

    active_set = np.arange(G.shape[0])
    k = 0
    while active_set.size > 0:
        # frontier = np.intersect1d(np.where(degrees == k)[0], active_set)
        rontier = np.array([v for v in active_set if degrees[v] == k]) 
        print(f"k = {k}")
        print(f'F: {frontier}')
        # print(active_set)
        print(f"D: {degrees}")
        while frontier.size > 0:
            for v in frontier:
                coreness[v] = k
                processed[v] = True  # Don't revisit it
            frontier, degrees = peel_fnc(frontier, k, G, degrees, processed)
        active_set = np.intersect1d(np.where(~processed)[0], active_set)
        k += 1
    return coreness

# def k_core_decomposition_julienne(G, peel_fnc, num_threads=1):
#     degrees = np.sum(G, axis=1) - 1
#     active_set = list(range(G.shape[0]))
#     coreness = np.zeros_like(degrees)
#     k = 0 
#     while list(active_set): 
#         frontier = np.intersect1d(np.argwhere(degrees == k), active_set)
#         # frontier = np.array([v for v in active_set if degrees[v] == k]) 
#         while list(frontier): 
#             for v in frontier: 
#                 coreness[v] = k
#             frontier, degrees = peel_fnc(frontier, k, G, degrees)
#         active_set = np.intersect1d(np.argwhere(degrees > k), active_set)
#         k += 1         
#     return coreness 

print("JULIENNE")
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
core_numbers = k_core_decomposition_julienne(adj, offline_peel, 1)
print(core_numbers)

H = nx.from_numpy_array(adj) #nx.havel_hakimi_graph(degrees)
solution = list(nx.core_number(H).values())
print(solution)

print((core_numbers == solution).all()) 