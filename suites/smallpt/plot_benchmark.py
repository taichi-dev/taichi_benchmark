import matplotlib.pyplot as plt
import sys
import os

from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda

cuda_sample_results = {
    'cuda_baseline': [{
        'spp': 16,
        'fps': 73
    }, {
        'spp': 32,
        'fps': 41
    }, {
        'spp': 48,
        'fps': 28
    }, {
        'spp': 64,
        'fps': 21
    }, {
        'spp': 80,
        'fps': 17
    }, {
        'spp': 96,
        'fps': 15
    }, {
        'spp': 112,
        'fps': 13
    }, {
        'spp': 128,
        'fps': 11
    }, {
        'spp': 144,
        'fps': 10
    }, {
        'spp': 160,
        'fps': 9
    }, {
        'spp': 176,
        'fps': 8
    }, {
        'spp': 192,
        'fps': 7
    }, {
        'spp': 208,
        'fps': 7
    }, {
        'spp': 224,
        'fps': 6
    }, {
        'spp': 240,
        'fps': 6
    }, {
        'spp': 256,
        'fps': 6
    }]
}

taichi_sample_results = {
    'taichi_baseline': [{
        'spp': 16,
        'fps': 81
    }, {
        'spp': 32,
        'fps': 42
    }, {
        'spp': 48,
        'fps': 27
    }, {
        'spp': 64,
        'fps': 21
    }, {
        'spp': 80,
        'fps': 16
    }, {
        'spp': 96,
        'fps': 14
    }, {
        'spp': 112,
        'fps': 12
    }, {
        'spp': 128,
        'fps': 10
    }, {
        'spp': 144,
        'fps': 9
    }, {
        'spp': 160,
        'fps': 8
    }, {
        'spp': 176,
        'fps': 7
    }, {
        'spp': 192,
        'fps': 7
    }, {
        'spp': 208,
        'fps': 6
    }, {
        'spp': 224,
        'fps': 6
    }, {
        'spp': 240,
        'fps': 5
    }, {
        'spp': 256,
        'fps': 5
    }]
}


def run_benchmarks():
    return benchmark_taichi(), benchmark_cuda()


def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["fps"])
    return perf


def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["spp"])
    return particles


def plot_line(cuda_results, taichi_results):
    plt.figure()
    x = extract_particles(taichi_results["taichi_baseline"])
    plt.plot(x, extract_perf(taichi_results["taichi_baseline"]), marker='o')
    plt.plot(x, extract_perf(cuda_results["cuda_baseline"]), marker='s')

    plt.grid('minor')
    plt.xlabel("Samples per pixel")
    plt.ylabel("Frames per Second")
    plt.legend(["Taichi", "CUDA"], loc='upper right')
    plt.title("Global Illumination Renderer (SmallPT)")
    plt.savefig("fig/bench.png", dpi=150)


if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        taichi_results = taichi_sample_results
        cuda_results = cuda_sample_results
    else:
        taichi_results, cuda_results = run_benchmarks()
    plot_line(cuda_results, taichi_results)
