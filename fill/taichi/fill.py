from time import perf_counter
import taichi as ti

ti.init(arch=ti.cuda)

#a = ti.field(float, 10000000)
a = ti.ndarray(float, 10000000)

a.fill(0.5)

t1_start = perf_counter()
for _ in range(500):
    a.fill(0.5)
ti.sync()
t1_stop = perf_counter()
print("Elapsed time ms:", (t1_stop-t1_start)*1000/500)
