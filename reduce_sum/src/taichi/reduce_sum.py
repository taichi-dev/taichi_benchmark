import taichi as ti 

ti.init(ti.cuda,
    kernel_profiler=True,
    )

def reduce_sum(nitems=2**26, repeats=5):
    f = ti.field(ti.f32, (nitems,))
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
    niter = 0
    while niter < repeats:
        init_kernel()
        ti.profiler.clear_kernel_profiler_info()
        reduce_sum_kernel()
        ti.sync()
        kernel_time += ti.profiler.get_kernel_profiler_total_time() * 1000
        niter += 1
    return kernel_time/repeats