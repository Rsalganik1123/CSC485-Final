import numpy as np 
import ipdb 

def all_neighbors(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()


def k_core_decomposition(G):
    # ipdb.set_trace()
    degrees = np.sum(G, axis=1) 
    active_set = list(range(G.shape[0]))
    coreness = np.zeros_like(degrees)
    
    def peel(frontier, k): 
        f_next = []
        for v in frontier:
            for u in all_neighbors(G, v): 
                degrees[u] =- 1 
                if degrees[u] == k: 
                    f_next.append(u)
        return f_next

    k = 0 
    while list(active_set): 
        frontier = np.intersect1d(np.argwhere(degrees == k), active_set)
        while list(frontier): 
            for v in frontier: 
                coreness[v] = k
            frontier = peel(frontier, k)
        active_set = np.intersect1d(np.argwhere(degrees > k), active_set)
        k += 1         
    return coreness 

def generate_random_adj_matrix(n, edge_prob=0.5, directed=False):
    # Generate upper triangle of random 0/1 values
    upper_triangle = np.triu(np.random.rand(n, n) < edge_prob, k=1).astype(int)
    adj_matrix = upper_triangle + upper_triangle.T

    return adj_matrix

# adj = generate_random_adj_matrix(5)
adj = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
degree = np.sum(adj, axis=1)
# print(degree)
core_numbers = k_core_decomposition(adj)
print(core_numbers)