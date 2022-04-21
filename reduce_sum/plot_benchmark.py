from matplotlib import pyplot as plt
from src.cuda.benchmark import benchmark as benchmark_cuda
from src.taichi.benchmark import benchmark as benchmark_taichi
import sys
import os

scale = [65536, 1048576, 16777216, 67108864]

def run_benchmarks():
    results = benchmark_cuda(scale)
    taichi_results = benchmark_taichi(scale)
    results['taichi'] = taichi_results['taichi']
    return results

def get_flops(ts):
    flops = []
    for ii in range(len(ts)):
        flops.append(4 * scale[ii] * 1000 / float(ts[ii]) / 1e9)
    return flops

def get_bandwidth(ts):
    Bandwidth = []
    for ii in range(len(ts)):
        Bandwidth.append(4 * scale[ii] * 1000 / float(ts[ii]) / 1e9)
    return Bandwidth

def plot_compute(results, machine="3080"):
    xlabel = ["256 KB","4 MB","64 MB", "256 MB"]
    fig, ax = plt.subplots()

    taichi_bandwidth = get_bandwidth(results['taichi'])
    bar_pos = [i*5+1 for i in range(len(taichi_bandwidth))]
    ax.bar(bar_pos, taichi_bandwidth)

    cuda_bandwidth = get_bandwidth(results['cuda'])
    bar_pos = [i*5+2 for i in range(len(cuda_bandwidth))]
    ax.bar(bar_pos, cuda_bandwidth)

    bar_pos = [i*5+2.5 for i in range(len(cuda_bandwidth))]
    ax.set_xticks(bar_pos, xlabel)

    cub_bandwidth = get_bandwidth(results['cub'])
    bar_pos = [i*5+3 for i in range(len(cub_bandwidth))]
    ax.bar(bar_pos, cub_bandwidth)
    
    thrust_bandwidth = get_bandwidth(results['thrust'])
    bar_pos = [i*5+4 for i in range(len(thrust_bandwidth))]
    ax.bar(bar_pos, thrust_bandwidth)

    ax.legend(['Taichi','CUDA', 'CUDA/cub', 'CUDA/thrust'])
    ax.set_xlabel("Data Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    if machine == "2060":
        plt.axhline(y = 336, color='grey', linestyle = 'dashed')
        plt.text(11, 336, 'DRAM Bandwidth=336GB/s')
    elif machine == "3080":
        plt.axhline(y = 760, color='grey', linestyle = 'dashed')
        plt.text(11, 770, 'DRAM Bandwidth=760GB/s')
    ax.set_title("ReduceSum Benchmark")
    plt.savefig(f"fig/compute_bench_{machine}.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass

    results = run_benchmarks()
    
    if len(sys.argv) == 2:
        plot_compute(results, sys.argv[1])
    else:
        plot_compute(results)