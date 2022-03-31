from time import perf_counter
import taichi as ti

def fill():
    ti.init(arch=ti.cuda)

    array_size = 1024 * 1024 * 24
    a = ti.ndarray(ti.float32, array_size)
    #a = ti.field(float, array_size)

    a.fill(0.5)

    t1_start = perf_counter()
    nIters = 10000
    for _ in range(nIters):
        a.fill(0.5)
    ti.sync()
    t1_stop = perf_counter()
    avg_time = (t1_stop-t1_start) * 1000 / nIters

print("Elapsed time ms:", avg_time)
print("Throughput: {} GB/s".format(1e-6 * array_size * 4 / avg_time))
print("Latency: {} ms".format(0))
