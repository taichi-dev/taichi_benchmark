import taichi as ti 

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
    while n_iter < repeats:
        init_kernel()
        ti.profiler.clear_kernel_profiler_info()
        reduce_sum_kernel()
        ti.sync()
        kernel_time += ti.profiler.get_kernel_profiler_total_time() * 1000
        n_iter += 1
    return kernel_time/repeats