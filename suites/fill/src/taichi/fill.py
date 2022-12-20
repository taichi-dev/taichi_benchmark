from time import perf_counter
import taichi as ti


def fill(N, nIter=500, printResults=False):
    ti.init(arch=ti.cuda,
        device_memory_fraction=0.9)

    a = ti.ndarray(ti.f32, N)

    a.fill(0.5)
    st = perf_counter()
    for _ in range(nIter):
        a.fill(0.5)
    ti.sync()
    et = perf_counter()
    wall_time = (et - st) * 1000.0 / nIter
    bandwidth = 1e-6 * N * 4 / wall_time

    if printResults:
        print("Time {}ms.".format(wall_time))
        print("Throughput: {} GB/s".format(bandwidth))

    return {'N' : N, 'time' : wall_time, 'bandwidth' : bandwidth}

if __name__ == '__main__':
    print(fill(2**30, printResults=True))
