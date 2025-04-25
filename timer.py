from datetime import datetime 
import networkx as nx 
import argparse 
import numpy as np 

from algorithms.paper_serial import k_core_decomposition_serial
from algorithms.my_nx import k_core_decomposition_nx
from algorithms.simplified_julienne import k_core_decomposition_julienne
from algorithms.pkc import k_core_decomposition_pkc


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['Karate', 'Cora', 'Citeseer'])
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--alg', choices=['nx_serial', 'paper_serial', 'julienne', 'pkc'])
    return parser.parse_args()

def main(): 
    
    args = parse_args() 
    print("DATASET", args.dataset)
    for alg in ['nx_serial', 'paper_serial', 'julienne', 'pkc']:
        print("*********")
        adj = np.load(f'/Users/rebeccasalganik/Documents/School/2025/Distributed/Data/{args.dataset}.npy').astype(int)
        G = nx.from_numpy_array(adj) 
        G.remove_edges_from(nx.selfloop_edges(G))
        solution = list(nx.core_number(G).values())

        args.alg = alg 
        if args.alg == 'nx_serial': method = k_core_decomposition_nx
        if args.alg == 'paper_serial': method = k_core_decomposition_serial
        if args.alg == 'julienne': method = k_core_decomposition_julienne
        if args.alg == 'pkc': method = k_core_decomposition_pkc 
        

        if alg == 'julienne' or alg == 'pkc': 
            for n in [1, 2, 4, 6, 8]: 
                args.num_threads = n
                times = [] 
                correct = [] 
                for i in range(10): 
                    b = datetime.now().timestamp()*1000
                    core_numbers = method(adj, args)
                    a = datetime.now().timestamp()*1000
                    times.append((a-b))
                    correct.append((core_numbers == solution)) 

                print(f'ALGORITHM: {alg}, THREAD_COUNT: {args.num_threads}, TIME:{np.mean(times)} STD:{np.std(times)}')
                    # print(len(times), np.mean(times), np.sum(correct)/len(times))
        else: 
            times = [] 
            correct = [] 
            for i in range(10): 
                b = datetime.now().timestamp()*1000
                core_numbers = method(adj, args)
                a = datetime.now().timestamp()*1000
                times.append(a-b)
                correct.append((core_numbers == solution)) 
            print(f'ALGORITHM: {alg}, THREAD_COUNT: {args.num_threads}, TIME:{np.mean(times)}, STD:{np.std(times)}')
            


        

main()