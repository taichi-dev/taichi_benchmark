import numpy as np
from smoke_taichi_cpu import run_smoke as run_smoke_cpu
from smoke_taichi_gpu import run_smoke as run_smoke_gpu

def benchmark():
    n_steps = np.arange(25, 200+25, 25).tolist()
    cpu_results = []
    gpu_results = []
    for n_step in n_steps:
        print("CPU running", "n_step", n_step)
        cpu_results.append(run_smoke_cpu(n_step))
        print("Done.")
        print("GPU running", "n_step", n_step)
        gpu_results.append(run_smoke_gpu(n_step))
        print("Done.")
    return {"taichi_cpu": cpu_results, "taichi_gpu": gpu_results}

if __name__ == "__main__":
    print(benchmark())
