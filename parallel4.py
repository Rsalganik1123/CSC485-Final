import numpy as np 
import ipdb 
import threading
from collections import Counter 

def nbrs_for_v(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()

def nbrs_all_gather(G, frontier): 
    # ipdb.set_trace() 
    return np.argwhere(G[frontier[0], :] == 1).flatten().tolist()



class MultiThreadLoader(object): 
    def __init__(self, output_path, run_function, platform=''): 
        self.output_path = output_path
        self.run_function = run_function
        self.fail_list = [] 
        self.success_count = 0 
        self.platform = platform
    def process(self, x, idx, **kwargs):
        batch_cnt = len(x)
        
        d = []
        with tqdm(desc=f'running thread: {idx}', unit='it', total=batch_cnt) as pbar:
            for i in x:
                try:
                    if self.run_function(i, self.output_path, **kwargs): 
                        d.append(i[0])
                        self.success_count += 1 
                except Exception as e:
                    self.fail_list.append(i)
                    print(f"[ERROR]: {e}", i)
                pbar.update() 
        return d



def peel(frontier, k, G, degrees): 
        f_next = []
        for v in frontier:
            for u in nbrs_for_v(G, v): 
                degrees[u] -= 1
                if degrees[u] == k: 
                    f_next.append(u)
        return f_next, degrees

def k_core_decomposition_serial(G):
    # ipdb.set_trace()
    degrees = np.sum(G, axis=1) 
    active_set = list(range(G.shape[0]))
    coreness = np.zeros_like(degrees)
    
    k = 0 
    while list(active_set): 
        frontier = np.intersect1d(np.argwhere(degrees == k), active_set)
        while list(frontier): 
            for v in frontier: 
                coreness[v] = k
            frontier, degrees = peel(frontier, k, G, degrees)
        active_set = np.intersect1d(np.argwhere(degrees > k), active_set)
        k += 1         
    return coreness 



def offline_peel_decr(H, idx, degrees, processed, k): 
    for u in idx:
        f_u = H[u]
        if not processed[u] and degrees[u] > k:
            degrees[u] = max(degrees[u] - f_u, 0)


def offline_peel(frontier, k, G, degrees, processed, num_threads=2):
    # Step 1: Gather all neighbors (with duplicates)
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
        frontier = np.intersect1d(np.where(degrees == k)[0], active_set)
        while frontier.size > 0:
            for v in frontier:
                coreness[v] = k
                processed[v] = True  # Don't revisit it
            frontier, degrees = peel_fnc(frontier, k, G, degrees, processed)
        active_set = np.intersect1d(np.where(~processed)[0], active_set)
        k += 1
    return coreness



# adj = generate_random_adj_matrix(5)
adj  = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
])
degree = np.sum(adj, axis=1)
core_numbers = k_core_decomposition_serial(adj)
print(core_numbers)
core_numbers = k_core_decomposition_julienne(adj, offline_peel, 1)
print(core_numbers)