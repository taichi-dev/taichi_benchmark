import cupy as cp
import time

def reduce_sum(n_items, repeats=5):
    f = cp.ones(shape=(n_items,), dtype=cp.float32)
    # f = cp.random.rand(n_items, dtype=cp.float32)    
    n_iter = 0
    sum = cp.sum(f)  # Skip the first run
    start = time.perf_counter()
    while n_iter < repeats:
        sum = cp.sum(f)
        n_iter += 1
    return (time.perf_counter() - start) * 1000 / repeats

def c_reduce_sum(n_items, repeats=5):
    '''
    User defined summation kernel;
    Achieved almost same perf but results were quirky
    '''
    reduce_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x',  # map
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '1.0',  # identity value
    'reduction'  # kernel name
    )
    x = cp.ones(shape=(n_items,), dtype=cp.float32)
    print(reduce_kernel(x, axis=0))

    n_iter = 0

    start = time.perf_counter()
    while n_iter < repeats:
        reduce_kernel(x, axis=0)
        n_iter += 1
    return (time.perf_counter() - start) * 1000 / repeats
