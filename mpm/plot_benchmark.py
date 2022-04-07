import matplotlib.pyplot as plt
import sys
import os

from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda

cuda_sample_results = {
    'cuda_2d': [{
        'n_particles': 512,
        'time_ms': 1.024374
    }, {
        'n_particles': 2048,
        'time_ms': 1.033906
    }, {
        'n_particles': 4608,
        'time_ms': 1.13101
    }, {
        'n_particles': 8192,
        'time_ms': 1.24747
    }, {
        'n_particles': 12800,
        'time_ms': 1.484127
    }, {
        'n_particles': 18432,
        'time_ms': 1.791808
    }, {
        'n_particles': 25088,
        'time_ms': 2.105247
    }, {
        'n_particles': 32768,
        'time_ms': 2.370462
    }],
    'cuda_3d': [{
        'n_particles': 8192,
        'time_ms': 2.70262
    }, {
        'n_particles': 65536,
        'time_ms': 8.443884
    }, {
        'n_particles': 221184,
        'time_ms': 36.317848
    }, {
        'n_particles': 524288,
        'time_ms': 174.325989
    }, {
        'n_particles': 1024000,
        'time_ms': 565.163757
    }, {
        'n_particles': 1769472,
        'time_ms': 1228.022095
    }, {
        'n_particles': 2809856,
        'time_ms': 2252.36377
    }, {
        'n_particles': 4194304,
        'time_ms': 3472.967773
    }]
}

taichi_sample_results = {
    'taichi_2d': [{
        'n_particles': 512,
        'time_ms': 0.4708201557548364
    }, {
        'n_particles': 2048,
        'time_ms': 0.4732567070391269
    }, {
        'n_particles': 4608,
        'time_ms': 0.5115689839101378
    }, {
        'n_particles': 8192,
        'time_ms': 0.5668949448249805
    }, {
        'n_particles': 12800,
        'time_ms': 0.6931407124000089
    }, {
        'n_particles': 18432,
        'time_ms': 0.8494079414163025
    }, {
        'n_particles': 25088,
        'time_ms': 1.0304002558427783
    }, {
        'n_particles': 32768,
        'time_ms': 1.2111213139576193
    }],
    'taichi_3d': [{
        'n_particles': 8192,
        'time_ms': 1.9398335703044722
    }, {
        'n_particles': 65536,
        'time_ms': 10.701219556153774
    }, {
        'n_particles': 221184,
        'time_ms': 34.867725920889825
    }, {
        'n_particles': 524288,
        'time_ms': 140.94088707861374
    }, {
        'n_particles': 1024000,
        'time_ms': 549.7929778569244
    }, {
        'n_particles': 1769472,
        'time_ms': 1286.6532287104633
    }, {
        'n_particles': 2809856,
        'time_ms': 2374.8291689565235
    }, {
        'n_particles': 4194304,
        'time_ms': 4005.1647448022436
    }]
}


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
    ax.bar(bar_pos, y_taichi, color='blue')

    bar_pos = [i * 3 + 1 for i in range(len(x_cuda))]
    ax.bar(bar_pos, y_cuda, color='green')

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
    plot_bar(cuda_results, taichi_results, "2d")
