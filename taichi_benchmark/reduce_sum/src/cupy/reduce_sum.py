import cupy as cp
import time
from cupyx.profiler import benchmark

def reduce_sum(n_items, repeats=5):
    '''
    Using CuPy's built-in GPU profiler for time measurement.
    It's common mistake to use Python's time.perf_counter()
    because CPU has no knowledge of GPU.
    See https://docs.cupy.dev/en/stable/user_guide/performance.html#benchmarking
    for further information.
    
    Use CUPY_ACCELERATORS=cub environment variable to switch
    to the CUB backend, which leads to a 100x performance boost.
    See https://docs.cupy.dev/en/stable/user_guide/performance.html
    for more details.
    '''
    f = cp.ones(shape=(n_items,), dtype=cp.float32)
    return benchmark(cp.sum, (f,), n_repeat=repeats).gpu_times.mean() * 1000

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
