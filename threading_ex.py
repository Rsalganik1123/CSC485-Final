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


def atomic_inc(a: atomics.INTEGRAL):
    res = atomics.CmpxchgResult(success=False, expected=a.load())
    while not res:
        desired = res.expected + 1
        res = a.cmpxchg_weak(expected=res.expected, desired=desired)
    return a.load() 

a = atomics.atomic(width=4, atype=atomics.INT)
a.store(5)


r = atomic_inc(a)
print(r)

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