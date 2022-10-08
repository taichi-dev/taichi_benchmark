from numba import cuda
import numpy as np
import time

def reduce_sum(n_items, repeats=5):
    f = np.ones(shape=(n_items,), dtype=np.float32)
    f_d = cuda.to_device(f)

    @cuda.reduce
    def reduce(a, b):
        return a + b

    reduce(f_d) # Skip first run

    n_iter = 0
    start = time.perf_counter()
    sum = 0.0
    while n_iter < repeats:
        sum = reduce(f_d)
        n_iter += 1
    return (time.perf_counter() - start)*1000 / repeats
