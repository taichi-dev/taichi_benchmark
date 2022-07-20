import taichi as ti 
import time

ti.init(ti.cuda,
    kernel_profiler=True,
    )

def reduce_sum(n_items, repeats=5):
    f = ti.field(ti.f32, (n_items,))
    @ti.kernel
    def init_kernel():
        for i in f:
            f[i] = 1.0
    @ti.kernel 
    def reduce_sum_kernel():
        sum = 0.0
        for i in ti.grouped(f):
            ti.atomic_add(sum, f[i])
    kernel_time = 0.0
    n_iter = 0
    init_kernel()
    reduce_sum_kernel()

    start = time.perf_counter()
    while n_iter < repeats:
        reduce_sum_kernel()
        ti.sync()
        n_iter += 1
    return ( time.perf_counter() - start ) * 1000 / repeats # *1000 to convert to msec
    # return kernel_time/repeats
