import taichi_benchmark as ti_bench

r_saxpy = ti_bench.saxpy.run()
print(r)
r_nbody = ti_bench.n_body.run()
print(r)
