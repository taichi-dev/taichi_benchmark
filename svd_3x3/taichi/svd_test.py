import taichi as ti
from time import perf_counter

ti.init(arch=ti.cuda, print_kernel_nvptx=True, device_memory_fraction=0.9, kernel_profiler=True)

def SVD(N, nest_factor=1):
    mat = ti.Matrix.field(3, 3, ti.f32, shape=(N,))
    res = ti.Matrix.field(3, 3, ti.f32, shape=(N * 3,))
    res2 = ti.Vector.field(3, ti.f32, shape=(N,))
    nIter = 10

    @ti.kernel
    def computeSVD():
        for I in ti.grouped(mat):
            U, S, V = ti.svd(mat[I])
            res[I * 2 + 0] = U
            res2[I] = [S[0, 0], S[0, 1], S[0, 2]]
            res[I * 2+ 1] = V

    computeSVD()
    st = perf_counter()
    for i in range(nIter):
        computeSVD()
    ti.sync()
    et = perf_counter()
    print((et - st) * 1000.0 / nest_factor / nIter)

SVD(1048576, 1)

ti.print_kernel_profile_info('trace')

