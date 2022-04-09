import matplotlib.pyplot as plt
import sys
import os

from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda


cuda_sample_results = {
    'cuda_baseline': [{
        'n_particles': 512,
        'time_ms': 0.335444
    }, {
        'n_particles': 2048,
        'time_ms': 0.320878
    }, {
        'n_particles': 4608,
        'time_ms': 0.450095
    }, {
        'n_particles': 8192,
        'time_ms': 0.472582
    }, {
        'n_particles': 12800,
        'time_ms': 0.572223
    }, {
        'n_particles': 18432,
        'time_ms': 0.822029
    }, {
        'n_particles': 25088,
        'time_ms': 1.023057
    }, {
        'n_particles': 32768,
        'time_ms': 1.22928
    }]
}

taichi_sample_results = {
    'taichi_baseline': [{
        'n_particles': 512,
        'time_ms': 0.5216462373027753
    }, {
        'n_particles': 2048,
        'time_ms': 0.5243271660049231
    }, {
        'n_particles': 4608,
        'time_ms': 0.546237979492048
    }, {
        'n_particles': 8192,
        'time_ms': 0.5765281875085293
    }, {
        'n_particles': 12800,
        'time_ms': 0.621946640620763
    }, {
        'n_particles': 18432,
        'time_ms': 0.8157364931662414
    }, {
        'n_particles': 25088,
        'time_ms': 0.9429942363254895
    }, {
        'n_particles': 32768,
        'time_ms': 1.0296167158259095
    }]
}


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


def plot_bar(cuda_results, taichi_results):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_series = "baseline"

    x_taichi = extract_particles(taichi_results["taichi_" + plot_series])
    y_taichi = extract_perf(taichi_results["taichi_" + plot_series])
    x_cuda = extract_particles(cuda_results["cuda_" + plot_series])
    y_cuda = extract_perf(cuda_results["cuda_" + plot_series])

    bar_pos = [i * 3 for i in range(len(x_cuda))]
    ax.bar(bar_pos, y_taichi)

    bar_pos = [i * 3 + 1 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_cuda)

    bar_pos_ticks = [i * 3 + 0.5 for i in range(len(x_cuda))]
    labels = ["{}".format(i) for i in x_cuda]
    ax.set_xticks(bar_pos_ticks, labels)

    plt.grid('minor', axis='y')
    plt.xlabel("Number of particles")
    plt.ylabel("Time per Frame (ms)")
    plt.legend(["Taichi", "CUDA"], loc='upper left')
    plt.title("Particle to Grid 2D (P2G)")
    plt.savefig("fig/bench.png", dpi=150)

def run_benchmarks():
    return benchmark_taichi(), benchmark_cuda()

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
    plot_bar(cuda_results, taichi_results)
