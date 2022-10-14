from taichi_benchmark.common import cuda_compile_and_benchmark
import os

def benchmark():
    workdir = os.path.dirname(os.path.abspath(__file__))
    return {"cublas": cuda_compile_and_benchmark(workdir, "cublas.cu", "saxpy_cublas", flags=['-lcublas']),
            "thrust": cuda_compile_and_benchmark(workdir, "thrust.cu", "saxpy_thrust")}