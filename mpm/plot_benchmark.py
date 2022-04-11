import matplotlib.pyplot as plt
import sys
import os

from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda

from misc import taichi_sample_results
from misc import cuda_sample_results

def run_benchmarks():
    return benchmark_taichi(), benchmark_cuda()


def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["time_ms"])
    return perf


def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["n_particles"])
    return particles


def plot_bar(cuda_results, taichi_results, plot_series="3d"):
    fig, ax = plt.subplots(figsize=(6, 4))

    x_cuda = extract_particles(cuda_results["cuda_" + plot_series])
    y_cuda = extract_perf(cuda_results["cuda_" + plot_series])
    x_taichi = extract_particles(taichi_results["taichi_" + plot_series])
    y_taichi = extract_perf(taichi_results["taichi_" + plot_series])

    bar_pos = [i * 3 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_taichi)

    bar_pos = [i * 3 + 1 for i in range(len(x_cuda))]
    ax.bar(bar_pos, y_cuda)

    bar_pos_ticks = [i * 3 + 0.5 for i in range(len(x_cuda))]
    labels = ["{}".format(i) for i in x_cuda]
    ax.set_xticks(bar_pos_ticks, labels)

    if plot_series == "3d":
        plt.yscale("log")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid('minor', axis='y')
    plt.xlabel("Number of particles")
    plt.ylabel("Time per Frame (ms)")
    plt.legend(["Taichi", "CUDA"], loc='upper left')
    plt.title("Material Point Method (" + plot_series.upper() + ")")
    plt.savefig("fig/bench_" + plot_series + ".png", dpi=150)


if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        cuda_results = cuda_sample_results
        taichi_results = taichi_sample_results
    else:
        taichi_results, cuda_results = run_benchmarks()
    plot_bar(cuda_results, taichi_results, "3d")
