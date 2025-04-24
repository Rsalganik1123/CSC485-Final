import numpy as np 
import ipdb 
import threading
from collections import Counter 
import atomics


def all_neighbors(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()

def atomic_dec(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res:
        desired = res.expected - 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired)
    return a.load() 

def online_peel_decr(frontier, G, atomic_array, k):
    f_next = [] 
    for v in frontier: 
        for u in all_neighbors(G, v): 
            delta = atomic_dec(atomic_array[u]) 
            if delta == k: 
                f_next.append(u) 
    return f_next 


def online_peel(frontier, k, G, atomic_array, degrees, num_threads=3): 
    f_next = [] 
    mt = num_threads > 1 
    if len(frontier) > num_threads: 
        if mt: 
            thread_list = [] 
            data_chunk = np.array_split(list(frontier), num_threads-1)
            for i,t in enumerate(data_chunk):
                m = threading.Thread(target=online_peel_decr, args=(frontier, G, atomic_array, k)) 
                thread_list.append(m)
            for m in thread_list:
                m.start() 
            for m in thread_list:
                m.join()
        else: 
            f_add = online_peel_decr(frontier, G, atomic_array, k)
            f_next.extend(f_add)
    else: 
        f_add = online_peel_decr(frontier, G, atomic_array, k)
        f_next.extend(f_add)
    for i in range(len(degrees)): 
        degrees[i] = atomic_array[i].load() 
    return f_next, degrees


def k_core_decomposition_pkc(G, peel_fnc, num_threads=1):
    degrees = np.sum(G, axis=1)
    atomic_array = [atomics.atomic(width=4, atype=atomics.INT) for d in degrees]
    for i,d in enumerate(degrees): 
        atomic_array[i].store(int(d)) 
    coreness = np.zeros_like(degrees)

    active_set = np.arange(G.shape[0])
    k = 0
    while list(active_set): 
        # frontier = np.intersect1d(np.argwhere(degrees == k), active_set)
        frontier = np.array([v for v in active_set if degrees[v] == k]) 
        while list(frontier): 
            for v in frontier: 
                coreness[v] = k
            frontier, degrees = peel_fnc(frontier, k, G, atomic_array, degrees)
        active_set = np.intersect1d(np.argwhere(degrees > k), active_set)
        k += 1           
    return coreness

# adj  = np.array([
#     [0, 1, 1, 0, 0, 0],
#     [1, 0, 1, 1, 0, 0],
#     [1, 1, 0, 0, 1, 0],
#     [0, 1, 0, 0, 1, 1],
#     [0, 0, 1, 1, 0, 1],
#     [0, 0, 0, 1, 1, 0]
# ])
adj = np.load('/Users/rebeccasalganik/Documents/School/2025/Distributed/Data/Karate.npy').astype(int)
degree = np.sum(adj, axis=1)
core_numbers = k_core_decomposition_pkc(adj, online_peel, 1)
print(core_numbers)

