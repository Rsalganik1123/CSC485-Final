# import atomics
# from threading import Thread


# def fn(ai: atomics.INTEGRAL, n: int):
#     for _ in range(n):
#         ai.inc()


# if __name__ == "__main__":
#     # setup
#     a = atomics.atomic(width=4, atype=atomics.INT)
#     total = 10000
    
#     # # # run threads to completion
#     t1 = Thread(target=fn, args=(a, 10000 // 2))
#     t2 = Thread(target=fn, args=(a, 10000 // 2))
#     t1.start(), t2.start()
#     t1.join(), t2.join()
#     # print results
#     print(f"a[{a.load()}] == total[{total}]")


# a = atomics.atomic(width=4, atype=atomics.INT)
# a.inc() 
# print(a.load())
# a.inc() 
# print(a.load())

import atomics
from ctypes import c_int
import numpy as np 
import threading

def atomic_inc(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res:
        desired = res.expected + 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired)
    return a.load() 

a = atomics.atomic(width=4, atype=atomics.INT)
# a.store(5)
# r = atomic_inc(a)
# print(r)


# def inc(degrees): 
#     degrees[2] = 55

# def main(): 
#     degrees = [1,2,3]
#     inc(degrees)
#     print(degrees)

# main() 




def atomic_dec(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res.success:
        desired = res.expected - 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired) 
    return desired

def online_peel_decr(frontier, atomic_array):
    f_next = [] 
    for v in frontier: 
        delta = atomic_dec(atomic_array[v]) 
        print("D", delta) 
    return f_next 


def test():
    a = atomics.atomic(width=4, atype=atomics.INT)
    a.store(5)
    r = atomic_inc(a)
    atomic_array = [atomics.atomic(width=4, atype=atomics.INT) for d in [5,4,3]]
    for j in range(len(atomic_array)):
        atomic_array[j].store(5) 
        print(atomic_array[j].load())
    data_chunk = np.array_split([1,1,1], 3)
    thread_list = [] 
    for i,t in enumerate(data_chunk):
        m = threading.Thread(target=online_peel_decr, args=(t, atomic_array)) 
        thread_list.append(m)
    for m in thread_list:
        m.start() 
    for m in thread_list:
        m.join()
    print("A", a.load())

test() 

# atomic_array = [atomics.atomic(width=4, atype=atomics.INT) for _ in range(10)]

# # Set values atomically
# atomic_array[0].store(42)
# atomic_array[1].store(100)

# # Read values atomically
# print(atomic_array[0].load())  # 42
# print(atomic_array[1].load())  # 100

# # Atomic increment
# atomic_array[0].fetch_add(1)
# print(atomic_array[0].load())  # 43

# import atomics
# a = atomics.atomic(width=4, atype=atomics.INT)
# a.store(5)

# # Try to change 5 → 10 if current value is 5
# success = a.compare_exchange(5, 10)