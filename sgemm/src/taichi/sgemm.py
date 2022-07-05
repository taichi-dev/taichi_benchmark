import taichi as ti
import numpy as np
import time

ti.init(arch=ti.cuda, device_memory_fraction=0.9)

def sgemm(M, K, N, test=False):
    NREPEAT = 1000

    @ti.kernel
    def impl(A: ti.template(), B: ti.template(), C: ti.template()):
        for m, n in C:
            c = 0.0
            for k in ti.static(range(K)):
                c += A[m, k] * B[k, n]
            C[m, n] = c

    A_ti = ti.field(shape=(M, K), dtype=ti.f32)
    B_ti = ti.field(shape=(N, K), dtype=ti.f32)
    C_ti = ti.field(shape=(M, N), dtype=ti.f32)

    A_np = np.random.random_sample((M, K)).astype(np.float32)
    B_np = np.random.random_sample((K, N)).astype(np.float32)

    A_ti.from_numpy(A_np)
    B_ti.from_numpy(B_np.T)

    # Initial run to cache shader and etc.
    impl(A_ti, B_ti, C_ti)

    # Measure latency.
    tic = time.perf_counter()
    for _ in range(NREPEAT):
        impl(A_ti, B_ti, C_ti)
    ti.sync()
    toc = time.perf_counter()
    dt = (toc - tic) / NREPEAT

    gflops = (2 * M * K * N * 1e-9) / dt
    gbs = (4 * (M * K + K * N + M * N) * 1e-9) / dt

    if test:
        C_np = A_np.dot(B_np)
        print("eps =", (C_ti.to_numpy() - C_np).mean())

    return (dt, gflops, gbs)

if __name__ == '__main__':
    for k in [256, 512, 1024, 2048, 4096]:
        for m in [256, 512, 1024, 2048, 4096]:
            for n in [256, 512, 1024, 2048, 4096]:
                dt, gflops, gbs = sgemm(m, k, n)
                print(f"M={m}, K={k}, N={n}, time={dt * 1e6} us, GFLOP/s={gflops}, GB/s={gbs}")
