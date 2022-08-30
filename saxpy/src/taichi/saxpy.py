import taichi as ti

ti.init(arch=ti.vulkan, device_memory_fraction=0.9)

def saxpy(N, len_coeff):
    x = ti.field(shape=(N, N), dtype=ti.float32)
    y = ti.field(shape=(N, N), dtype=ti.float32)

    coeff = []
    for ind in range(len_coeff):
        i = ind + 1
        coeff.append(2 * i * i * i / 3.1415926535)

    @ti.kernel
    def saxpy_kernel(x: ti.template(), y: ti.template()):
        for i in ti.grouped(y):
            # Statically unroll
            z_c = x[i]
            for c in ti.static(coeff):
                z_c = c * z_c + y[i]
            y[i] = z_c

    @ti.kernel
    def init(x: ti.template()):
        for i in ti.grouped(x):
            x[i] = ti.random(ti.float32) * 64.0

    def run(repeats=5000):
        import time

        init(x)
        init(y)
        saxpy_kernel(x, y)

        st = time.perf_counter()
        for i in range(repeats):
            saxpy_kernel(x, y)
        ti.sync()
        et = time.perf_counter()
        avg_time = (et - st) / repeats
        GFlops = 1e-9 * len(coeff) * 2 * N * N / avg_time
        GBs = 1e-9 * N * N * 4 * 3 / avg_time
        return {"N":N, "fold":len_coeff, "time":avg_time*1000.0, "gflops":GFlops,"gbs":GBs}
    return run()

if __name__ == '__main__':
    for i in [256, 512, 1024, 2048, 4096]:
        for j in range(16):
            rd = saxpy(i, 2**(j))
            print("{}x{}@{}, {:.3f}ms, {:.3f} GFLOPS, {:.3f} GB/s".format(rd["N"], rd["N"], rd["fold"], rd["time"], rd["gflops"], rd["gbs"]))
