from datetime import datetime 

from paper_serial import k_core_decomposition_serial
from my_nx import k_core_decomposition_nx
from julienne import k_core_decomposition_julienne
from pkc import k_core_decomposition_pkc


def main(): 
    dataset = 'Karate'
    adj = np.load(f'/Users/rebeccasalganik/Documents/School/2025/Distributed/Data/{dataset}.npy').astype(int)

    H = nx.from_numpy_array(adj) #nx.havel_hakimi_graph(degrees)
# solution = list(nx.core_number(H).values())

# print((core_numbers == solution).all()) 