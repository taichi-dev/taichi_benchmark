import matplotlib.pyplot as plt
import sys
import os

cuda_sample_results = {'cuda_baseline': [{'n_particles': 512, 'time_ms': 0.444458}, {'n_particles': 2048, 'time_ms': 0.44377}, {'n_particles': 4608, 'time_ms': 0.481411}, {'n_particles': 8192, 'time_ms': 0.487567}, {'n_particles': 12800, 'time_ms': 0.551323}, {'n_particles': 18432, 'time_ms': 0.673213}, {'n_particles': 25088, 'time_ms': 0.746727}, {'n_particles': 32768, 'time_ms': 0.871354}]}

taichi_sample_results = {'taichi_baseline': [{'n_particles': 512, 'time_ms': 0.3269685341784623}, {'n_particles': 2048, 'time_ms': 0.32648004296831346}, {'n_particles': 4608, 'time_ms': 0.3284883232410607}, {'n_particles': 8192, 'time_ms': 0.33818339550961696}, {'n_particles': 12800, 'time_ms': 0.33936923242094963}, {'n_particles': 18432, 'time_ms': 0.36472302343426577}, {'n_particles': 25088, 'time_ms': 0.3777477568362997}, {'n_particles': 32768, 'time_ms': 0.38578442383041534}]}

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
    fig, ax = plt.subplots(figsize=(12,9))
    plot_series = "baseline"

    x_cuda = extract_particles(cuda_results["cuda_" + plot_series])
    y_cuda = extract_perf(cuda_results["cuda_" + plot_series])
    bar_pos = [i*3 for i in range(len(x_cuda))]
    ax.bar(bar_pos, y_cuda)

    x_taichi = extract_particles(taichi_results["taichi_" + plot_series])
    y_taichi = extract_perf(taichi_results["taichi_" + plot_series])
    bar_pos = [i*3+1 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_taichi)

    labels = ["{}".format(i) for i in x_cuda]
    ax.set_xticks(bar_pos, labels, rotation = 30)
    
    plt.grid('minor', axis='y')
    plt.xlabel("#Particles")
    plt.ylabel("Time per Frame (ms)")
    plt.legend(["CUDA", "Taichi"], loc='upper left')
    plt.title("P2G benchmark")
    plt.savefig("fig/bench.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    cuda_results = cuda_sample_results
    taichi_results = taichi_sample_results
    plot_bar(cuda_results, taichi_results)
