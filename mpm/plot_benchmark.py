from benchmark_cuda import benchmark as benchmark_cuda
from benchmark_taichi import benchmark as benchmark_taichi

import matplotlib.pyplot as plt
import sys
import os


cuda_sample_results = {'taichi_baseline': [{'n_particles': 16, 'time': 0.7686792123990926}, {'n_particles': 128, 'time': 0.7934009423777866}, {'n_particles': 1024, 'time': 0.833564200192427}, {'n_particles': 8192, 'time': 1.8873389345728242}, {'n_particles': 65536, 'time': 10.569774629885842}]}

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

def plot(cuda_results, taichi_results, plot_cuda_roofline=True):
    plt.figure()
    x = extract_nbodies(taichi_results["taichi_baseline"])
    plt.plot(x, extract_perf(taichi_results["taichi_baseline"]), marker='o')
    plt.plot(x, extract_perf(taichi_results["taichi_block"]), marker='o')
    plt.plot(x, extract_perf(taichi_results["taichi_unroll"]), marker='o')
    plt.plot(x, extract_perf(cuda_results["cuda_baseline"]), marker='P', ms=6)
    plt.plot(x, extract_perf(cuda_results["cuda_block"]), marker='P', ms=6)
    if plot_cuda_roofline:
        plt.plot(x, extract_perf(cuda_results["cuda_best"]), marker='P', ms=6)
    plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("#Bodies")
    plt.ylabel("Speed (billion body interactions per second)")
    plt.legend(["Taichi/Baseline", "Taichi/Block", "Taichi/Unroll", "CUDA/Baseline", "CUDA/Block", "CUDA/Roofline"], loc='lower right')
    plt.title("N-Body benchmark")
    if plot_cuda_roofline:
        plt.savefig("fig/bench_roofline.png", dpi=150)
    else:
        plt.savefig("fig/bench.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        cuda_results = cuda_sample_results
        #taichi_results = taichi_sample_results
    else:
        cuda_results, taichi_results = run_benchmarks()
    #plot(cuda_results, taichi_results, plot_cuda_roofline=True)
    plot(cuda_results, taichi_results, plot_cuda_roofline=False)
