from .taichi.benchmark import benchmark as bench_taichi
from .cuda.benchmark import benchmark as bench_cuda

def run():
    ti_res = bench_taichi(max_nesting=128)
    return ti_res
