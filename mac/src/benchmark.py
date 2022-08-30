from numba_kernels import launch_numba, launch_numba_cuda, calc_velocity_numba
from taichi_kernels import launch_taichi, init_taichi
from utils import gen_data_numpy, gen_data_numba_cuda, gen_data_taichi, compute_ground_truth, nx, ny

from multiprocessing import Process
import time


def run(gen_data_func, calc_func, name, ut_groundtruth=None, vt_groundtruth=None, ctx_init_func=None):
    print(f"=========================================")
    print(f"Running benchmark for target {name}...")
    if ctx_init_func is not None:
        ctx_init_func()
    data_args = gen_data_func()
    calc_func(*data_args, name=name,
              ut_gt=ut_groundtruth, vt_gt=vt_groundtruth)


print(f"Benchmark for {nx}x{ny} arrays")
print("Computing ground truth...")
ut, vt = compute_ground_truth(calc_velocity_numba)
print("Done")

# Benchmark Numba CPU
p = run(gen_data_numpy, launch_numba, "Numba-CPU")

# Benchmark Numba CUDA
# Numpy CUDA has context conflict with Taichi, move into a subprocess.
p = Process(target=run, args=(gen_data_numba_cuda,
            launch_numba_cuda, "Numba-CUDA", ut, vt))
p.start()
p.join()
time.sleep(1)

# Benchmark Taichi CUDA
p = Process(target=run, args=(gen_data_taichi, launch_taichi,
            "Taichi-CUDA", ut, vt, init_taichi))
p.start()
p.join()
