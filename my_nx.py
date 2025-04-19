import numpy as np 
import ipdb 

def all_neighbors(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()

def k_core_decomposition(G, k=None):
    degree_counts = np.sum(G, axis=1).tolist() 
    degrees = dict(zip(list(range(G.shape[0])), degree_counts)) 
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: all_neighbors(G, v) for v in nodes}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core 

def generate_random_adj_matrix(n, edge_prob=0.5, directed=False):
    # Generate upper triangle of random 0/1 values
    upper_triangle = np.triu(np.random.rand(n, n) < edge_prob, k=1).astype(int)
    adj_matrix = upper_triangle + upper_triangle.T

    return adj_matrix

# adj = generate_random_adj_matrix(5)
# print(adj)
adj = np.array([[0, 0, 0, 0, 1],
 [0, 0, 0, 1, 0],
 [0, 0,  0,  1, 1],
 [0, 1, 1, 0, 0],
 [1, 0, 1, 0, 0]])
degree = np.sum(adj, axis=1)
print(degree)
core_numbers = k_core_decomposition(adj)
print(core_numbers)