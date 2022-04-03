import taichi as ti
from time import perf_counter

ti.init(arch=ti.cuda,
        print_kernel_nvptx=True,
        device_memory_fraction=0.9,
        kernel_profiler=True)

def SVD(N, nIter=1):
    mat = ti.field(ti.f32)
    res = ti.field(ti.f32)

    ti.root.dense(ti.ij, (3, 3)).dense(ti.k, N).place(mat)
    ti.root.dense(ti.ij, (21, N)).place(res)

    ti_mat3 = ti.types.matrix(3, 3, ti.f32)
    ti_vec3 = ti.types.vector(3, ti.f32)
    res_vec_t = ti.types.vector(21, ti.f32)

    @ti.func
    def fill_result_vector(i:ti.i32, U:ti.template(), S:ti.template(), V:ti.template()):
        for x, y in ti.static(ti.ndrange(3, 3)):
            res[x * 3 + y, i] = U[x, y]
        for x in ti.static(range(3)):
            res[9 + x, i] = S[x, x]
        for x, y in ti.static(ti.ndrange(3, 3)):
            res[12 + x * 3 + y, i] = V[x, y]

    @ti.func
    def load_input_vector(i:ti.i32, pad:ti.template()) :
        for x, y in ti.static(ti.ndrange(3, 3)):
            pad[x, y] = mat[x, y, i]
            #mat[i][x, y]

    @ti.kernel
    def computeSVD():
        for i in range(N):
            pad = ti_mat3(0.0)
            load_input_vector(i, pad)
            U, S, V = ti.svd(pad)
            fill_result_vector(i, U, S, V)

    def benchmark():
        # Warm-up run
        computeSVD()

        # Wall clock
        st = perf_counter()
        # Kernel time
        ti.profiler.clear_kernel_profiler_info()
        for i in range(nIter):
            computeSVD()
        ti.sync()
        # Stop wall clock
        et = perf_counter()
        # Get kernel time
        kernel_time = ti.profiler.get_kernel_profiler_total_time() * 1000.0 / nIter
        wall_time = (et - st) * 1000.0 / nIter
        # Result display
        print("Kernel average time {}ms".format(kernel_time))
        print("Python scope wall average time {}ms".format(wall_time))
        ti.profiler.print_kernel_profiler_info('trace')
        return {"N":N, "kernel_time": kernel_time, "wall_time":wall_time}
    return benchmark()

if __name__ == '__main__':
    SVD(1048576, 10)
    #SVD(16, 10)
