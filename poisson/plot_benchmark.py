import matplotlib.pyplot as plt
import sys
import os

numba_sample_results = {'numba_gpu': [{'desired_samples': 1000, 'time_ms': 17.34950498212129}, {'desired_samples': 5000, 'time_ms': 107.88116841576993}, {'desired_samples': 10000, 'time_ms': 226.49038201197982}, {'desired_samples': 50000, 'time_ms': 1260.7922151917592}, {'desired_samples': 100000, 'time_ms': 1344.771138811484}]}

numpy_sample_results = {'numpy_cpu': [{'desired_samples': 1000, 'time_ms': 991.2736028200015}, {'desired_samples': 5000, 'time_ms': 5998.027631384321}, {'desired_samples': 10000, 'time_ms': 12623.283750191331}, {'desired_samples': 50000, 'time_ms': 70075.99696719553}, {'desired_samples': 100000, 'time_ms': 74836.31602639798}]}

taichi_sample_results = {'taichi_cpu': [{'desired_samples': 1000, 'time_ms': 14.330414799042046}, {'desired_samples': 5000, 'time_ms': 69.50833240989596}, {'desired_samples': 10000, 'time_ms': 138.9469366054982}, {'desired_samples': 50000, 'time_ms': 691.8933660024777}, {'desired_samples': 100000, 'time_ms': 755.0804344005883}]}

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["time_ms"])
    return perf

def extract_samples(results):
    samples = []
    for record in results:
        samples.append(record["desired_samples"])
    return samples

def plot_bar(numpy_results, numba_results, taichi_results):
    fig, ax = plt.subplots(figsize=(6, 4))

    x_taichi = extract_samples(taichi_results["taichi_cpu"])
    y_taichi = extract_perf(taichi_results["taichi_cpu"])
    x_numba = extract_samples(numba_results["numba_gpu"])
    y_numba = extract_perf(numba_results["numba_gpu"])
    x_numpy = extract_samples(numpy_results["numpy_cpu"])
    y_numpy = extract_perf(numpy_results["numpy_cpu"])

    labels = ["{}".format(i) for i in x_taichi]

    # series
    bar_pos = [i * 4 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_taichi, color='blue')

    bar_pos = [i * 4 + 1 for i in range(len(x_numba))]
    ax.bar(bar_pos, y_numba, color='orange')

    bar_pos = [i * 4 + 2 for i in range(len(x_numpy))]
    ax.bar(bar_pos, y_numpy, color='green')

    bar_pos_ticks = [i * 4 + 1 for i in range(len(x_taichi))]
    ax.set_xticks(bar_pos_ticks, labels)

    plt.yscale("log")  
    plt.grid('minor', axis='y')
    plt.xlabel("Samples")
    plt.ylabel("Execution time in ms")
    plt.legend([
        "Taichi",
        "Numba",
        "Numpy",
    ],
               loc='upper left')
    plt.title("Poisson Disk Sampling")
    plt.savefig("fig/bench.png", dpi=150)


if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    plot_bar(numpy_sample_results, numba_sample_results, taichi_sample_results)
