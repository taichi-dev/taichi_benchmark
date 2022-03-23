import matplotlib.pyplot as plt
import sys
import os

cuda_sample_results = {'cuda_baseline': [{'spp': 16, 'fps': 73}, {'spp': 32, 'fps': 41}, {'spp': 48, 'fps': 28}, {'spp': 64, 'fps': 21}, {'spp': 80, 'fps': 17}, {'spp': 96, 'fps': 15}, {'spp': 112, 'fps': 13}, {'spp': 128, 'fps': 11}, {'spp': 144, 'fps': 10}, {'spp': 160, 'fps': 9}, {'spp': 176, 'fps': 8}, {'spp': 192, 'fps': 7}, {'spp': 208, 'fps': 7}, {'spp': 224, 'fps': 6}, {'spp': 240, 'fps': 6}, {'spp': 256, 'fps': 6}]}

#cuda_sample_results = {'cuda_baseline': [{'spp': 32, 'time_ms': 24.336449}, {'spp': 64, 'time_ms': 46.224194}, {'spp': 96, 'time_ms': 68.754845}, {'spp': 128, 'time_ms': 84.639229}, {'spp': 160, 'time_ms': 105.246147}, {'spp': 192, 'time_ms': 127.489891}, {'spp': 224, 'time_ms': 146.096603}, {'spp': 256, 'time_ms': 166.291687}]}

taichi_sample_results = {'taichi_baseline': [{'spp': 16, 'fps': 81}, {'spp': 32, 'fps': 42}, {'spp': 48, 'fps': 27}, {'spp': 64, 'fps': 21}, {'spp': 80, 'fps': 16}, {'spp': 96, 'fps': 14}, {'spp': 112, 'fps': 12}, {'spp': 128, 'fps': 10}, {'spp': 144, 'fps': 9}, {'spp': 160, 'fps': 8}, {'spp': 176, 'fps': 7}, {'spp': 192, 'fps': 7}, {'spp': 208, 'fps': 6}, {'spp': 224, 'fps': 6}, {'spp': 240, 'fps': 5}, {'spp': 256, 'fps': 5}]}

#taichi_sample_results = {'taichi_baseline': [{'spp': 32, 'time_ms': 24.63049694779329}, {'spp': 64, 'time_ms': 49.856646699481644}, {'spp': 96, 'time_ms': 78.17163055005949}, {'spp': 128, 'time_ms': 99.65819650096819}, {'spp': 160, 'time_ms': 125.85285250097513}, {'spp': 192, 'time_ms': 149.39746184973046}, {'spp': 224, 'time_ms': 175.58412284997758}, {'spp': 256, 'time_ms': 201.95406079874374}]}


def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["fps"])
        #perf.append(record["time_ms"])
    return perf

def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["spp"])
    return particles 

def plot_line(cuda_results, taichi_results):
    plt.figure()
    x = extract_particles(cuda_results["cuda_baseline"])
    plt.plot(x, extract_perf(cuda_results["cuda_baseline"]), marker='s')
    plt.plot(x, extract_perf(taichi_results["taichi_baseline"]), marker='o')

    #plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("Samples per pixel")
    plt.ylabel("Frames per Second")
    plt.legend(["CUDA", "Taichi"], loc='upper right')
    plt.title("SMALLPT benchmark")
    plt.savefig("fig/bench.png", dpi=150)

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
    
    if plot_series == "3d":
        plt.yscale("log")  
    plt.grid('minor', axis='y')
    plt.xlabel("#Particles")
    plt.ylabel("Time per Frame (ms)")
    plt.legend(["CUDA", "Taichi"], loc='upper left')
    plt.title("SmallPT benchmark")
    plt.savefig("fig/bench_" + plot_series + ".png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    cuda_results = cuda_sample_results
    taichi_results = taichi_sample_results
    #plot_line(cuda_results, taichi_results)
    plot_bar(cuda_results, taichi_results)
