import numpy as np 
import ipdb 
# import dgl 

def all_neighbors(G, v): 
    return np.argwhere(G[v, :] == 1).flatten().tolist()

def k_core_decomposition_nx(G, args):
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
    return list(core.values()) 