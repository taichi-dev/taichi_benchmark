import numpy as np
import time

def reduce_sum(n_items, repeats=5):
    f = np.ones(shape=(n_items,), dtype=np.float32)
    n_iter = 0
    start = time.perf_counter()
    sum = 0.0
    while n_iter < repeats:
        sum = np.sum(f)
        n_iter += 1
        # print(sum) Verified results are correct
    return (time.perf_counter() - start)*1000 / repeats
