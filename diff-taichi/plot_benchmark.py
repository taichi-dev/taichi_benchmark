import matplotlib.pyplot as plt
import sys
import os

jax_cpu_results = {'jax_cpu': [{'steps': 25, 'time_ms': 65.24398803710938}, {'steps': 50, 'time_ms': 130.02352905273438}, {'steps': 75, 'time_ms': 194.32464599609375}, {'steps': 100, 'time_ms': 259.0186767578125}, {'steps': 125, 'time_ms': 323.78143310546875}, {'steps': 150, 'time_ms': 388.3748779296875}, {'steps': 175, 'time_ms': 452.9609680175781}, {'steps': 200, 'time_ms': 516.94384765625}]}

jax_gpu_results = {'jax_gpu': [{'steps': 25, 'time_ms': 10.312166213989258}, {'steps': 50, 'time_ms': 18.36493682861328}, {'steps': 75, 'time_ms': 28.193653106689453}, {'steps': 100, 'time_ms': 35.162445068359375}, {'steps': 125, 'time_ms': 43.2327995300293}, {'steps': 150, 'time_ms': 52.01408386230469}, {'steps': 175, 'time_ms': 59.739013671875}, {'steps': 200, 'time_ms': 68.09099578857422}]}

taichi_results = {'taichi_cpu': [{'steps': 25, 'time_ms': 51.112630011630245}, {'steps': 50, 'time_ms': 104.13628999958746}, {'steps': 75, 'time_ms': 159.63488050329033}, {'steps': 100, 'time_ms': 210.41491949290503}, {'steps': 125, 'time_ms': 261.4528949925443}, {'steps': 150, 'time_ms': 319.54752300225664}, {'steps': 175, 'time_ms': 371.35305399715435}, {'steps': 200, 'time_ms': 422.6975054916693}], 'taichi_gpu': [{'steps': 25, 'time_ms': 14.285370503785089}, {'steps': 50, 'time_ms': 28.69464700052049}, {'steps': 75, 'time_ms': 43.09935700439382}, {'steps': 100, 'time_ms': 57.2914744989248}, {'steps': 125, 'time_ms': 71.7793994845124}, {'steps': 150, 'time_ms': 86.17233450058848}, {'steps': 175, 'time_ms': 100.47039449273143}, {'steps': 200, 'time_ms': 113.97033199318685}]}


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

def plot_bar(cuda_results, taichi_results):
    fig, ax = plt.subplots(figsize=(12,9))

    plot_series = "gpu"
    x_cuda = extract_particles(cuda_results["jax_" + plot_series])
    y_cuda = extract_perf(cuda_results["jax_" + plot_series])
    bar_pos = [i*3 for i in range(len(x_cuda))]
    ax.bar(bar_pos, y_cuda)

    x_taichi = extract_particles(taichi_results["taichi_" + plot_series])
    y_taichi = extract_perf(taichi_results["taichi_" + plot_series])
    bar_pos = [i*3+1 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_taichi)

    labels = ["{}".format(i) for i in x_cuda]
    ax.set_xticks(bar_pos, labels)
    
    plt.grid('minor', axis='y')
    plt.xlabel("#Steps")
    plt.ylabel("Time (ms)")
    plt.legend(["CUDA", "Taichi"], loc='upper left')
    plt.title("DiffTaichi benchmark")
    plt.savefig("fig/bench_"+ plot_series + ".png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass

    cuda_results = {**jax_cpu_results, **jax_gpu_results}
    taichi_results = taichi_results 
    plot_bar(cuda_results, taichi_results)
