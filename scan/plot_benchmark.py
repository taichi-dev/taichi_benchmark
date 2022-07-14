from matplotlib import pyplot as plt
from src.cuda.benchmark import benchmark as benchmark_cuda
import sys
import os

scale = [65536, 1048576, 16777216, 67108864]

def run_benchmarks():
    results = benchmark_cuda(scale)
    return results

def get_bandwidth(ts):
    Bandwidth = []
    for ii in range(len(ts)):
        Bandwidth.append(8 * scale[ii] * 1000 / float(ts[ii]) / 1e9)
    return Bandwidth

def plot_compute(results, machine="2060"):
    xlabel = ["2**16","2**20","2**24", "2**26"]
    fig, ax = plt.subplots()

    cuda_bandwidth = get_bandwidth(results['cuda'])
    bar_pos = [i*4+0 for i in range(len(cuda_bandwidth))]
    ax.bar(bar_pos, cuda_bandwidth)

    cub_bandwidth = get_bandwidth(results['cub'])
    bar_pos = [i*4+1 for i in range(len(cub_bandwidth))]
    ax.bar(bar_pos, cub_bandwidth)
    ax.set_xticks(bar_pos, xlabel)

    thrust_bandwidth = get_bandwidth(results['thrust'])
    bar_pos = [i*4+2 for i in range(len(thrust_bandwidth))]
    ax.bar(bar_pos, thrust_bandwidth)

    ax.legend(['CUDA/CUDA', 'CUDA/cub', 'CUDA/thrust'])
    ax.set_xlabel("Array shape")
    ax.set_ylabel("Bandwidth (GB/s)")
    if machine == "2060":
        plt.axhline(y = 336, color='grey', linestyle = 'dashed')
        plt.text(11, 336, 'DRAM Bandwidth=336GB/s')
    elif machine == "3080":
        plt.axhline(y = 760, color='grey', linestyle = 'dashed')
        plt.text(11, 770/6.0, 'DRAM Bandwidth=760GB/s')
    ax.set_title("ReduceSum benchmark on 1D arrays")
    plt.savefig("fig/compute_bench.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass

    results = run_benchmarks()
    plot_compute(results)