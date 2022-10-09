from .cuda.benchmark import benchmark as benchmark_cuda
from .taichi.benchmark import benchmark as benchmark_taichi

import matplotlib.pyplot as plt
import sys
import os

from misc import cuda_sample_results
from misc import taichi_sample_results

def run_benchmarks():
    return benchmark_cuda(), benchmark_taichi()

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["rate"])
    return perf

def extract_nbodies(results):
    nbodies = []
    for record in results:
        nbodies.append(record["nbodies"])
    return nbodies

def plot(cuda_results, taichi_results):
    plt.figure()
    x = extract_nbodies(taichi_results["taichi_baseline"])
    plt.plot(x, extract_perf(taichi_results["taichi_baseline"]), marker='o')
    plt.plot(x, extract_perf(taichi_results["taichi_cache_block"]), marker='o')
    plt.plot(x, extract_perf(cuda_results["cuda_baseline"]), marker='P', ms=6)
    plt.plot(x, extract_perf(cuda_results["cuda_block"]), marker='P', ms=6)
    plt.plot(x, extract_perf(cuda_results["cuda_best"]), marker='P', ms=6)
    plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("#Bodies")
    plt.ylabel("Speed (billion body interactions per second)")
    plt.legend(["Taichi/Baseline", "Taichi/CacheBlock", "CUDA/Baseline", "CUDA/Block", "CUDA/Opt"], loc='lower right')
    plt.title("N-Body benchmark")
    plt.savefig("fig/bench.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        cuda_results = cuda_sample_results
        taichi_results = taichi_sample_results
    else:
        cuda_results, taichi_results = run_benchmarks()
    plot(cuda_results, taichi_results)
