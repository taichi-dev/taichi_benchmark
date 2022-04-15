from matplotlib import pyplot as plt
from src.cuda.benchmark import benchmark as benchmark_cuda
import sys
import os

scale = [65536, 1048576, 16777216, 67108864]

def run_benchmarks():
    return benchmark_cuda(scale)

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

def plot_compute(results, machine="2060"):
    xlabel = ["2**16","2**20","2**24", "2**26"]
    fig, ax = plt.subplots()

    cuda_flops = get_flops(results['cuda'])
    bar_pos = [i*5+1 for i in range(len(cuda_flops))]
    ax.bar(bar_pos, cuda_flops)
    ax.set_xticks(bar_pos, xlabel)

    cub_flops = get_flops(results['cub'])
    bar_pos = [i*5+2 for i in range(len(cub_flops))]
    ax.bar(bar_pos, cub_flops)
    ax.set_xticks(bar_pos, xlabel)
    
    thrust_flops = get_flops(results['thrust'])
    bar_pos = [i*5+3 for i in range(len(thrust_flops))]
    ax.bar(bar_pos, thrust_flops)
    ax.set_xticks(bar_pos, xlabel)

    ax.legend(['CUDA/CUDA', 'CUDA/cub', 'CUDA/thrust'])
    ax.set_xlabel("Array shape")
    ax.set_ylabel("Performance (GFLOPS)")
    def comp2mem(x):
        return x
    def mem2comp(x):
        return x
    ax2 = ax.secondary_yaxis('right', functions=(comp2mem, mem2comp))
    ax2.set_ylabel("Bandwidth (GB/s)")
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