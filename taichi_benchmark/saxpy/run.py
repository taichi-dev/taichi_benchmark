from .taichi.benchmark import benchmark as bench_taichi
from .cuda.benchmark import benchmark as bench_cuda
import math
import os

def run():
    ti_res = bench_taichi(max_nesting=128)
    cuda_res = bench_cuda()
    print(ti_res)
    print(cuda_res)
