import numpy as np 
# import ipdb 
from collections import Counter 
import threading
import atomics


def all_neighbors(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()



def atomic_dec(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res:
        desired = res.expected - 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired)
    return a.load() 

def atomic_inc(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res:
        desired = res.expected + 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired)
    return a.load() 


class NodeSampleVal:
    def __init__(): 
        self.mode = False #bool flag for whether sampling mode is on 
        self.rate = 0.0 #rate for sampling v
        self.cnt = 0 #number of hits in the sampling process 







class SamplingStruct: 
    def __init__(self, G, threshold): 
        # self.G = G
        self.degrees = np.sum(G, axis=1)
        # self.sampler_array = [NodeSampleVal for d in self.degrees]
        self.atomic_array = [atomics.atomic(width=4, atype=atomics.INT) for d in self.degrees]
        for i,d in enumerate(self.degrees): 
            self.atomic_array[i].store(int(d)) 
        
    
    def validate(self, v,k): 
        return ((self.degrees[v] * r) > k) or (self.sampler_array[v].cnt < self.sampler_array[v].rate * ((self.degrees[v]-k))/k)
    
    def resample(self, v,k,frontier):
        self.degrees[v] = np.intersect1d(self.active_set, all_neighbors(G, v))
        if self.degrees[v] <= k: 
            frontier.append(v)
        self.set_sampler(v,k)

    def set_sampler(self, v, k):
        if (((self.degrees[v] * r) > k) or (self.degrees[v] > self.threshold)): 
            self.sampler_array[v].mode = True 
            self.sampler_array[v].rate = mu/((1-r)*self.degrees[v])
            self.sampler_array[v].cnt = 0 
        else: 
            self.sampler_array[v].mode = False

    def online_peel(self, frontier, k, atomic_array): 
        f_next = [] 
        self.c = [] 
        for v in frontier: 
            for u in all_neighbors(self.G, v): 
                if self.sampler_array[u].mode: 
                    delta = atomic_inc(sampler_array[u].cnt) #add probability later 
                    if delta == self.mu -1: 
                        self.c.append(u)
                    else: 
                        delta = atomic_dec(atomic_array[u])
                        if delta == k: 
                            f_next.apend(u)
        for i in range(len(self.degrees)): 
            self.degrees[i] = atomic_array[i].load() 
        return f_next, self.degrees

    def k_core_decomposition_paper(self, G): 
        
        coreness = np.zeros_like(degrees)

        active_set = np.arange(G.shape[0])
        while list(self.active_set): 
            frontier = np.array([v for v in self.active_set if self.degrees[v] == k]) 
            for v in range(len(degrees)): 
                if self.sampler_array[v].mode: 
                    if not self.validate(v, k): 
                        self.resample(v,k,frontier)
            while list(frontier): 
                for v in frontier: 
                    coreness[v] = k
                    frontier, self.c = self.online_peel(frontier,k)
                for v in self.c: 
                    self.resample(v,k, frontier)
            self.active_set = np.intersect1d(np.argwhere(self.degrees > k), self.active_set)
            k += 1




adj  = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
])


r = .1 



# degree = np.sum(adj, axis=1)
runner = SamplingStruct(adj, 5)


# print(core_numbers)

