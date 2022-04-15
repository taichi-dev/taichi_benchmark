import taichi as ti 

ti.init(ti.cuda, kernel_profiler=True)

@ti.kernel 
def ReduceSum():
    f = ti.field(ti.f32, (2 ** 22,))
    sum = 0.0
    # Kernel time
    ti.profiler.clear_kernel_profiler_info()
    for i in ti.grouped(f):
        ti.atomic_add(sum, f[i])
    ti.sync()
    kernel_time = ti.profiler.get_kernel_profiler_total_time() * 1000.0
    print(sum)
    print(kernel_time)
    #return kernel_time

reduce()