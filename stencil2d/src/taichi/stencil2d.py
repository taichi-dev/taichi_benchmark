import taichi as ti
import time


def run_stencil(N = 512, arch=ti.vulkan, bench=True):
    ti.init(arch=arch)
    neighbours = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    @ti.kernel
    def init(x: ti.template()):
        for I in ti.grouped(x):
            if (I[0] == 0) or (I[0] == N - 1):
                x[I] = 1.0
            else:
                x[I] = 0.0

    @ti.kernel
    def jacobi_step(x: ti.template(), y: ti.template()):
        for I in ti.grouped(x):
            if (I[0] == 0) or (I[1] == 0) or (I[0] == N - 1) or (I[1] == N - 1):
                y[I] = x[I]
            else:
                p = 0.0
                for stencil in ti.static(neighbours):
                    p += x[I + stencil]
                y[I] = p * 0.25

        for I in ti.grouped(x):
            if (I[0] == 0) or (I[1] == 0) or (I[0] == N - 1) or (I[1] == N - 1):
                x[I] = y[I]
            else:
                p = 0.0
                for stencil in ti.static(neighbours):
                    p += y[I + stencil]
                x[I] = p * 0.25

    def benchmark(bench_iter=10000):
        x = ti.field(shape=(N, N), dtype=ti.float32)
        y = ti.field(shape=(N, N), dtype=ti.float32)
        init(x)
        jacobi_step(x, y)
        jacobi_step(y, x)
        st = time.time()
        for i in range(bench_iter):
            jacobi_step(x, y)
            #jacobi_step(y, x)
        ti.sync()
        et = time.time()
        avg_time_ms = (et - st) * 1000.0 / float(bench_iter)  / 2.0 #two steps per iter
        Gflops = 1e-6 * N * N * (len(neighbours) + 1) / avg_time_ms
        GBs = 1e-6 * N * N * 4 * 2.0 / avg_time_ms
        return {"N":N, "time":avg_time_ms, "gflops":Gflops, "gbs":GBs}

    def visualize():
        x = ti.field(shape=(N, N), dtype=ti.float32)
        y = ti.field(shape=(N, N), dtype=ti.float32)
        gui = ti.GUI('Stencil test', (N, N))
        init(x)
        steps = 0
        while gui.running:
            nIter = 50
            for i in range(nIter):
                jacobi_step(x, y)
                jacobi_step(y, x)
            steps += nIter
            if steps > 20000:
                init(x)
                steps = 0
            gui.set_image(x)
            gui.show()

    if bench:
        return benchmark()
    else:
        visualize()

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == "visual":
        run_stencil(512, bench=False)
    else:
        k = 64
        for i in range(8):
            res = run_stencil(k)
            print("Stencil {}x{}, {:.3f} GFLOPS, {:.3f} GB/s".format(res["N"],res["N"], res["gflops"], res["gbs"]))
            k *= 2
