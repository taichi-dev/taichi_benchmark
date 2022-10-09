import taichi as ti
from taichi_benchmark.common import benchmark

@benchmark(test_name='Nested SAXPY', archs=[ti.cuda, ti.opengl, ti.vulkan, ti.metal])
def saxpy(**kwargs):
    N = kwargs['N']
    len_coeff = kwargs['len_coeff']

    x = ti.field(shape=(N, N), dtype=ti.float32)
    y = ti.field(shape=(N, N), dtype=ti.float32)

    coeff = []
    for ind in range(len_coeff):
        i = ind + 1
        coeff.append(2 * i * i * i / 3.1415926535)

    @ti.kernel
    def init(x: ti.template()):
        for i in ti.grouped(x):
            x[i] = ti.random(ti.float32) * 64.0

    @ti.kernel
    def saxpy_kernel(x: ti.template(), y: ti.template()):
        for i in ti.grouped(y):
            # Statically unroll
            z_c = x[i]
            for c in ti.static(coeff):
                z_c = c * z_c + y[i]
            y[i] = z_c

    def benchmark_init(): 
        init(x)
        init(y)
        saxpy_kernel(x, y)
    
    def benchmark_iter():
        saxpy_kernel(x, y)

    def benchmark_metrics(avg_time):
        gflops = 1e-9 * len(coeff) * 2 * N * N / avg_time
        gbs =  1e-9 * N * N * 4 * 3 / avg_time
        return {'GFLOPS': gflops, 'GB/s': gbs}

    return benchmark_init, benchmark_iter, benchmark_metrics