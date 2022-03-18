#from benchmark_cuda import benchmark as benchmark_cuda
#from benchmark_taichi import benchmark as benchmark_taichi

import matplotlib.pyplot as plt
import sys
import os

#cuda_sample_results = {'cuda_baseline': [{'n_particles': 128, 'time': 0.632}, {'n_particles': 512, 'time': 0.628}, {'n_particles': 2048, 'time': 0.647}, {'n_particles': 8192, 'time': 0.773}, {'n_particles': 32768, 'time': 2.352}]}
cuda_sample_results = {'cuda_baseline': [{'n_particles': 128, 'time': 0.636}, {'n_particles': 512, 'time': 0.637}, {'n_particles': 1152, 'time': 0.643}, {'n_particles': 2048, 'time': 0.645}, {'n_particles': 3200, 'time': 0.659}, {'n_particles': 4608, 'time': 0.702}, {'n_particles': 6272, 'time': 0.718}, {'n_particles': 8192, 'time': 0.772}, {'n_particles': 10368, 'time': 1.353}, {'n_particles': 12800, 'time': 1.462}, {'n_particles': 15488, 'time': 1.587}, {'n_particles': 18432, 'time': 1.796}, {'n_particles': 21632, 'time': 1.798}, {'n_particles': 25088, 'time': 2.092}, {'n_particles': 28800, 'time': 2.23}, {'n_particles': 32768, 'time': 2.488}]}


taichi_sample_results = {'taichi_baseline': [{'n_particles': 128, 'time': 0.3175147197254091}, {'n_particles': 512, 'time': 0.3185752607421932}, {'n_particles': 1152, 'time': 0.3198986733394804}, {'n_particles': 2048, 'time': 0.31859193944683284}, {'n_particles': 3200, 'time': 0.32409828710910915}, {'n_particles': 4608, 'time': 0.3323211162111761}, {'n_particles': 6272, 'time': 0.3408275234377811}, {'n_particles': 8192, 'time': 0.368112465331194}, {'n_particles': 10368, 'time': 0.6006463120158401}, {'n_particles': 12800, 'time': 0.6319817875990452}, {'n_particles': 15488, 'time': 0.6921672548827473}, {'n_particles': 18432, 'time': 0.8786921074204201}, {'n_particles': 21632, 'time': 0.9603567587888051}, {'n_particles': 25088, 'time': 1.0176723403318988}, {'n_particles': 28800, 'time': 1.1343457509767063}, {'n_particles': 32768, 'time': 1.2218140800754895}]}

#taichi_sample_results = {'taichi_baseline': [{'n_particles': 128, 'time': 0.3188079111353659}, {'n_particles': 512, 'time': 0.31917445996043625}, {'n_particles': 2048, 'time': 0.3207612705082852}, {'n_particles': 8192, 'time': 0.3716928662171881}, {'n_particles': 32768, 'time': 1.2241718540053625}]}

def run_benchmarks():
    return benchmark_cuda(), benchmark_taichi()

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["time"])
    return perf

def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["n_particles"])
    return particles 

def plot(cuda_results, taichi_results, plot_cuda_roofline=True):
    plt.figure()
    x = extract_particles(cuda_results["cuda_baseline"])

    plt.plot(x, extract_perf(cuda_results["cuda_baseline"]), marker='P')
    plt.plot(x, extract_perf(taichi_results["taichi_baseline"]), marker='o')
    #if plot_cuda_roofline:
    #    plt.plot(x, extract_perf(cuda_results["cuda_best"]), marker='P', ms=6)

    plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("#Particles")
    plt.ylabel("Speed (millisecond)")
    plt.legend(["CUDA", "Taichi"], loc='lower right')
    plt.title("MPM benchmark")
    if plot_cuda_roofline:
        plt.savefig("fig/bench_roofline.png", dpi=150)
    else:
        plt.savefig("fig/bench.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    #if len(sys.argv) >= 2 and sys.argv[1] == "sample":
    cuda_results = cuda_sample_results
    taichi_results = taichi_sample_results
    #else:
    #    cuda_results, taichi_results = run_benchmarks()
    #plot(cuda_results, taichi_results, plot_cuda_roofline=True)
    plot(cuda_results, taichi_results, plot_cuda_roofline=False)
