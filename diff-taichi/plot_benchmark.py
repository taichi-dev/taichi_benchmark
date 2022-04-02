import matplotlib.pyplot as plt
import sys
import os

from src.taichi.benchmark import benchmark as benchmark_taichi
from src.jax.benchmark import benchmark as benchmark_jax

jax_cpu_results = {
    'jax_cpu': [{
        'steps': 25,
        'time_ms': 65.91094207763672
    }, {
        'steps': 50,
        'time_ms': 131.53125
    }, {
        'steps': 75,
        'time_ms': 196.27066040039062
    }, {
        'steps': 100,
        'time_ms': 261.52276611328125
    }, {
        'steps': 125,
        'time_ms': 326.4784240722656
    }, {
        'steps': 150,
        'time_ms': 391.44482421875
    }, {
        'steps': 175,
        'time_ms': 456.5045166015625
    }, {
        'steps': 200,
        'time_ms': 522.333251953125
    }]
}

jax_gpu_results = {
    'jax_gpu': [{
        'steps': 25,
        'time_ms': 10.054882049560547
    }, {
        'steps': 50,
        'time_ms': 20.362703323364258
    }, {
        'steps': 75,
        'time_ms': 28.36370849609375
    }, {
        'steps': 100,
        'time_ms': 37.734405517578125
    }, {
        'steps': 125,
        'time_ms': 49.2039909362793
    }, {
        'steps': 150,
        'time_ms': 53.763267517089844
    }, {
        'steps': 175,
        'time_ms': 66.99272918701172
    }, {
        'steps': 200,
        'time_ms': 71.54842376708984
    }]
}

taichi_results = {
    'taichi_cpu': [{
        'steps': 25,
        'time_ms': 51.08319049759302
    }, {
        'steps': 50,
        'time_ms': 103.85819099610671
    }, {
        'steps': 75,
        'time_ms': 158.39250650606118
    }, {
        'steps': 100,
        'time_ms': 209.04668950242922
    }, {
        'steps': 125,
        'time_ms': 259.88954899366945
    }, {
        'steps': 150,
        'time_ms': 318.44818698300514
    }, {
        'steps': 175,
        'time_ms': 368.9125679957215
    }, {
        'steps': 200,
        'time_ms': 418.57923500356264
    }],
    'taichi_gpu': [{
        'steps': 25,
        'time_ms': 4.84518900339026
    }, {
        'steps': 50,
        'time_ms': 9.666016005212441
    }, {
        'steps': 75,
        'time_ms': 14.41958149371203
    }, {
        'steps': 100,
        'time_ms': 19.13971251633484
    }, {
        'steps': 125,
        'time_ms': 23.946640998474322
    }, {
        'steps': 150,
        'time_ms': 28.695989021798596
    }, {
        'steps': 175,
        'time_ms': 33.55165199900512
    }, {
        'steps': 200,
        'time_ms': 38.40553350164555
    }]
}


def run_benchmarks():
    return benchmark_taichi(), benchmark_jax()

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["time_ms"])
    return perf


def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["steps"])
    return particles


def plot_bar(jax_results, taichi_results, plot_series):
    fig, ax = plt.subplots(figsize=(6, 4))

    x_jax = extract_particles(jax_results["jax_" + plot_series])
    y_jax = extract_perf(jax_results["jax_" + plot_series])
    x_taichi = extract_particles(taichi_results["taichi_" + plot_series])
    y_taichi = extract_perf(taichi_results["taichi_" + plot_series])

    labels = ["{}".format(i) for i in x_jax]

    # series
    bar_pos = [i * 3 for i in range(len(x_jax))]
    ax.bar(bar_pos, y_taichi, color='deepskyblue')

    bar_pos = [i * 3 + 1 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_jax, color='orange')

    bar_pos_ticks = [i * 3 + 0.5 for i in range(len(x_taichi))]
    ax.set_xticks(bar_pos_ticks, labels)

    plt.grid('minor', axis='y')
    plt.xlabel("Simulation steps")
    plt.ylabel("Execution time in ms")
    plt.legend([
        "Taichi (" + plot_series.upper() + ")",
        "JAX (" + plot_series.upper() + ")"
    ],
               loc='upper left')
    plt.title("Differentiable Incompressible Fluid Simulator (Smoke)")
    plt.savefig("fig/bench_" + plot_series + ".png", dpi=150)


if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass

    plot_series = "cpu" # choose to plot cpu or gpu
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        taichi_results = taichi_results
        jax_results = {**jax_cpu_results, **jax_gpu_results}
    else:
        if plot_series == "cpu":
            taichi_results, jax_cpu_results = run_benchmarks()
        else:
            taichi_results, jax_gpu_results = run_benchmarks()

        jax_results = {**jax_cpu_results, **jax_gpu_results}
    plot_bar(jax_results, taichi_results, plot_series)
