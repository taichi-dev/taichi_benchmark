import numpy as np
from numba import njit, prange
import time

def reduce_sum(n_items, repeats=5):
    f = np.ones(shape=(n_items,), dtype=np.float32)
    n_iter = 0
    
    @njit(parallel=True, fastmath=True)
    def reduce():
        sum = 0.0
        for i in prange(n_items):
            sum += f[i]
        return sum

    reduce() # skip first run to avoid compilation time

    sum = 0.0        
    start = time.perf_counter()
    while n_iter < repeats:
        sum = reduce()
        n_iter += 1

    return (time.perf_counter() - start) * 1000 / repeats
