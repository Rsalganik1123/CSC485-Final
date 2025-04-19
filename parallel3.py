import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

NUM_ROUND = 5


class Graph:
    def __init__(self, n, adj_matrix):
        self.n = n  # Number of nodes
        self.adj_matrix = adj_matrix  # Adjacency matrix (numpy array)

    def get_neighbors(self, v):
        """Returns the neighbors of node v as a list."""
        return np.nonzero(self.adj_matrix[v])[0]


def pal_verifier(G, act_core):
    n = G.n
    num_buckets = 16
    buckets = [set() for _ in range(num_buckets)]
    frontier = []
    exp_core = np.zeros(n, dtype=int)

    # Initial core numbers (degree of each node)
    for i in range(n):
        exp_core[i] = np.sum(G.adj_matrix[i])  # Degree of node i

    max_deg = np.max(exp_core)
    max_core = 0

    for k in range(0, max_deg + 1, num_buckets):
        max_deg = 0
        for i in range(n):
            max_deg = max(max_deg, exp_core[i])
            if k <= exp_core[i] < k + num_buckets:
                buckets[exp_core[i] - k].add(i)

        if max_deg < k:
            break

        # Process nodes in the buckets
        for i in range(num_buckets):
            # Use a list to safely collect the nodes to process
            to_process = list(buckets[i])
            while to_process:
                for j in to_process:
                    if exp_core[j] == k + i:
                        max_core = max(max_core, k + i)
                        neighbors = G.get_neighbors(j)
                        for v in neighbors:
                            if exp_core[v] > k + i:
                                exp_core[v] -= 1
                                if exp_core[v] >= k + i and exp_core[v] < k + num_buckets:
                                    buckets[exp_core[v] - k].add(v)

                # Clear and repopulate the bucket with the next frontier
                to_process = list(buckets[i])
                buckets[i].clear()

    # Verifying core numbers
    for i in range(n):
        if exp_core[i] != act_core[i]:
            print(f"exp_core[{i}]: {exp_core[i]} while act_core[{i}]: {act_core[i]}")
            # assert exp_core[i] == act_core[i]


def k_core_decomposition(G):
    n = G.n
    degrees = np.sum(G.adj_matrix, axis=1)  # Degree of each node
    coreness = np.zeros_like(degrees)

    active_set = list(range(n))
    k = 0

    while active_set:
        frontier = [i for i in active_set if degrees[i] == k]

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(frontier)):
                futures.append(executor.submit(peel, G, frontier[i], k, degrees, coreness))

            # Collect results from the threads
            for future in futures:
                result = future.result()
                active_set = list(set(active_set) & set(result))

        k += 1

    return coreness


def peel(G, v, k, degrees, coreness):
    f_next = []
    neighbors = G.get_neighbors(v)
    for u in neighbors:
        if degrees[u] > k:
            degrees[u] -= 1
            if degrees[u] == k:
                f_next.append(u)

    coreness[v] = k
    return f_next


def run(KCore, G, verify):
    total_time = 0
    coreness = np.zeros(G.n, dtype=int)
    for i in range(NUM_ROUND + 1):
        start_time = time.time()
        coreness = KCore(G)
        end_time = time.time()
        if i == 0:
            print(f"Warmup Round: {end_time - start_time}")
        else:
            print(f"Round {i}: {end_time - start_time}")
            total_time += (end_time - start_time)

    average_time = total_time / NUM_ROUND
    print(f"Average time: {average_time}")

    if verify:
        print("Running verifier...")
        start_time = time.time()
        coreness = KCore(G)
        print(coreness)
        end_time = time.time()
        pal_verifier(G, coreness)

    with open("kcore.tsv", "a") as f:
        f.write(f"{average_time}\n")


def main(input_path=None, symmetrized=False, verify=False):
    # Example graph: Number of nodes
    n = 5

    # Adjacency matrix representation (example)
    adj_matrix = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])

    G = Graph(n, adj_matrix)

    print(f"Running on graph: |V|={G.n}, |E|={np.sum(G.adj_matrix) // 2}")

    run(k_core_decomposition, G, verify)


# Example usage
if __name__ == "__main__":
    main(verify=True)
