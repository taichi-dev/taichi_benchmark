from .src.taichi.benchmark import benchmark as bench_taichi
from .src.cuda.benchmark import benchmark as bench_cuda
import math
import os

def run():
    ti_res = bench_taichi(max_nesting=128)
    print(ti_res)
    # cuda_res = bench_cuda()
    # print(cuda_res)