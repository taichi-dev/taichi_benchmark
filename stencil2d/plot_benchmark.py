from benchmark_cuda import benchmark as benchmark_cuda
from benchmark_taichi import benchmark as benchmark_taichi

import matplotlib.pyplot as plt
import sys
import os


cuda_sample_results={'cuda': [{'N': 64, 'time': 0.002, 'gflops': 13.653, 'gbs': 21.845}, {'N': 128, 'time': 0.002, 'gflops': 40.96, 'gbs': 65.536}, {'N': 256, 'time': 0.002, 'gflops': 163.84, 'gbs': 262.144}, {'N': 512, 'time': 0.004, 'gflops': 374.491, 'gbs': 599.186}, {'N': 1024, 'time': 0.014, 'gflops': 374.491, 'gbs': 599.186}, {'N': 2048, 'time': 0.051, 'gflops': 415.278, 'gbs': 664.444}, {'N': 4096, 'time': 0.195, 'gflops': 431.291, 'gbs': 690.065}, {'N': 8192, 'time': 0.768, 'gflops': 436.622, 'gbs': 698.596}]}
taichi_sample_results={'taichi': [{'N': 64, 'time': 0.0032423615455627443, 'gflops': 6.316383818463246, 'gbs': 10.106214109541193}, {'N': 128, 'time': 0.0032091617584228517, 'gflops': 25.52691517808056, 'gbs': 40.8430642849289}, {'N': 256, 'time': 0.0032439112663269045, 'gflops': 101.01386045957832, 'gbs': 161.62217673532533}, {'N': 512, 'time': 0.003414595127105713, 'gflops': 383.85810065599065, 'gbs': 614.1729610495851}, {'N': 1024, 'time': 0.014106535911560058, 'gflops': 371.6631803066231, 'gbs': 594.661088490597}, {'N': 2048, 'time': 0.0495990514755249, 'gflops': 422.82098903340085, 'gbs': 676.5135824534414}, {'N': 4096, 'time': 0.19068535566329955, 'gflops': 439.9188375436699, 'gbs': 703.8701400698719}, {'N': 8192, 'time': 0.755373752117157, 'gflops': 444.2096631760614, 'gbs': 710.7354610816982}]}

def run_benchmarks():
    return benchmark_cuda(), benchmark_taichi()

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["gbs"])
    return perf

def extract_N(results):
    N = []
    for record in results:
        N.append(record["N"])
    return N

def plot(cuda_results, taichi_results):
    plt.figure()
    x = extract_N(taichi_results["taichi"])
    plt.plot(x, extract_perf(taichi_results["taichi"]), marker='o')
    plt.plot(x, extract_perf(cuda_results["cuda"]), marker='P', ms=6)
    plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("Matrix dimensions")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend(["Taichi", "CUDA"], loc='lower right')
    plt.title("Stencil benchmark")
    plt.axhline(y = 760, color='grey', linestyle = 'dashed')
    plt.text(80, 730, 'DRAM Bandwidth=760GB/s')
    plt.axvline(x = 723, color='grey', linestyle = 'dashed')
    plt.text(723, 200, 'L2 Cache=4MB', rotation=270)
    plt.savefig("fig/bench.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        cuda_results = cuda_sample_results
        taichi_results = taichi_sample_results
    else:
        cuda_results, taichi_results = run_benchmarks()
    print(cuda_results, taichi_results)
    plot(cuda_results, taichi_results)
