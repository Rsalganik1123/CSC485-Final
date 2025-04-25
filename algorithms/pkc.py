import numpy as np 
import ipdb 
import threading
from collections import Counter 
import atomics
import networkx as nx 

def all_neighbors(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()

def atomic_dec(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res:
        desired = res.expected - 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired)
    return desired #res #a.load() 

def online_peel_decr(frontier, G, atomic_array, k, f_next):
    for v in frontier: 
        for u in all_neighbors(G, v): 
            delta = atomic_dec(atomic_array[u]) 
            if delta == k:
                f_next.extend([u]) 
    return f_next 


def online_peel(frontier, k, G, atomic_array, degrees, num_threads): 
    f_next = []  
    mt = num_threads > 1  
    if mt: 
        thread_list = [] 
        data_chunk = np.array_split(list(frontier), num_threads)
        for i,t in enumerate(data_chunk):
            m = threading.Thread(target=online_peel_decr, args=(t, G, atomic_array, k, f_next)) 
            thread_list.append(m)
        for m in thread_list:
            m.start() 
        for m in thread_list:
            m.join()
    else: 
        f_next = online_peel_decr(frontier, G, atomic_array, k, f_next)
    for i in range(len(degrees)): 
        degrees[i] = atomic_array[i].load() 
    return f_next, degrees

def k_core_decomposition_pkc(G, args):
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
            frontier, degrees = online_peel(frontier, k, G, atomic_array, degrees, args.num_threads)
        active_set = np.intersect1d(np.argwhere(degrees > k), active_set)
        k += 1           
    return coreness.tolist() 