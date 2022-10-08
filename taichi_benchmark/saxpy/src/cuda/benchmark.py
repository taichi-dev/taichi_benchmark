from taichi_benchmark.common import cuda_compile_and_benchmark

def benchmark():
    return {"cublas": cuda_compile_and_benchmark("cublas.cu", "saxpy_cublas", flags=['-lcublas']),
            "thrust": cuda_compile_and_benchmark("thrust.cu", "saxpy_thrust")}