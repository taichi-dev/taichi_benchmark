from matplotlib import pyplot as plt
from benchmark_taichi import benchmark_taichi
from benchmark_cuda import benchmark as benchmark_cuda
import sys
import os

sample_res = '{"cublas": [{"N": 256, "time": 0.002, "gflops": 58.671, "gbs": 352.029}, {"N": 512, "time": 0.003, "gflops": 170.522, "gbs": 1023.134}, {"N": 1024, "time": 0.021, "gflops": 101.976, "gbs": 611.855}, {"N": 2048, "time": 0.075, "gflops": 111.558, "gbs": 669.347}, {"N": 4096, "time": 0.293, "gflops": 114.517, "gbs": 687.102}], "thrust": [{"N": 256, "fold": 1, "time": 0.006, "gflops": 23.711, "gbs": 142.269}, {"N": 256, "fold": 2, "time": 0.005, "gflops": 47.977, "gbs": 143.93}, {"N": 256, "fold": 4, "time": 0.006, "gflops": 94.805, "gbs": 142.207}, {"N": 256, "fold": 8, "time": 0.006, "gflops": 184.875, "gbs": 138.657}, {"N": 256, "fold": 16, "time": 0.006, "gflops": 334.826, "gbs": 125.56}, {"N": 512, "fold": 1, "time": 0.006, "gflops": 81.535, "gbs": 489.212}, {"N": 512, "fold": 2, "time": 0.006, "gflops": 163.743, "gbs": 491.228}, {"N": 512, "fold": 4, "time": 0.006, "gflops": 325.564, "gbs": 488.346}, {"N": 512, "fold": 8, "time": 0.007, "gflops": 641.9, "gbs": 481.425}, {"N": 512, "fold": 16, "time": 0.007, "gflops": 1188.255, "gbs": 445.596}, {"N": 1024, "fold": 1, "time": 0.024, "gflops": 88.949, "gbs": 533.694}, {"N": 1024, "fold": 2, "time": 0.024, "gflops": 178.07, "gbs": 534.211}, {"N": 1024, "fold": 4, "time": 0.023, "gflops": 357.312, "gbs": 535.968}, {"N": 1024, "fold": 8, "time": 0.024, "gflops": 709.66, "gbs": 532.245}, {"N": 1024, "fold": 16, "time": 0.024, "gflops": 1392.069, "gbs": 522.026}, {"N": 2048, "fold": 1, "time": 0.079, "gflops": 106.381, "gbs": 638.284}, {"N": 2048, "fold": 2, "time": 0.079, "gflops": 212.751, "gbs": 638.252}, {"N": 2048, "fold": 4, "time": 0.079, "gflops": 425.298, "gbs": 637.948}, {"N": 2048, "fold": 8, "time": 0.079, "gflops": 849.912, "gbs": 637.434}, {"N": 2048, "fold": 16, "time": 0.079, "gflops": 1689.668, "gbs": 633.625}, {"N": 4096, "fold": 1, "time": 0.296, "gflops": 113.187, "gbs": 679.122}, {"N": 4096, "fold": 2, "time": 0.296, "gflops": 226.36, "gbs": 679.079}, {"N": 4096, "fold": 4, "time": 0.296, "gflops": 452.762, "gbs": 679.144}, {"N": 4096, "fold": 8, "time": 0.297, "gflops": 905.013, "gbs": 678.76}, {"N": 4096, "fold": 16, "time": 0.297, "gflops": 1807.705, "gbs": 677.889}], "taichi": [{"N": 256, "fold": 1, "time": 0.006319954199716448, "gflops": 20.739390802211936, "gbs": 124.43634481327163}, {"N": 256, "fold": 2, "time": 0.0062749893870204685, "gflops": 41.77600691122012, "gbs": 125.32802073366038}, {"N": 256, "fold": 4, "time": 0.006239291187375783, "gflops": 84.03005794325061, "gbs": 126.04508691487594}, {"N": 256, "fold": 8, "time": 0.006320460978895426, "gflops": 165.90182322164276, "gbs": 124.42636741623207}, {"N": 256, "fold": 16, "time": 0.0063150141853839156, "gflops": 332.0898320155564, "gbs": 124.53368700583364}, {"N": 512, "fold": 1, "time": 0.006552474806085229, "gflops": 80.0137376359079, "gbs": 480.08242581544744}, {"N": 512, "fold": 2, "time": 0.00637908037751913, "gflops": 164.37729859861065, "gbs": 493.131895795832}, {"N": 512, "fold": 4, "time": 0.006395104434341192, "gflops": 327.9308448410106, "gbs": 491.8962672615159}, {"N": 512, "fold": 8, "time": 0.006447763834148646, "gflops": 650.5052151237501, "gbs": 487.8789113428126}, {"N": 512, "fold": 16, "time": 0.006529038399457931, "gflops": 1284.815234154, "gbs": 481.8057128077501}, {"N": 1024, "fold": 1, "time": 0.02107226480729878, "gflops": 99.52190802355575, "gbs": 597.1314481413345}, {"N": 1024, "fold": 2, "time": 0.021460041217505933, "gflops": 195.4471549000808, "gbs": 586.3414647002425}, {"N": 1024, "fold": 4, "time": 0.020669475989416243, "gflops": 405.8452185384558, "gbs": 608.7678278076838}, {"N": 1024, "fold": 8, "time": 0.02099389820359647, "gflops": 799.1472492291065, "gbs": 599.3604369218299}, {"N": 1024, "fold": 16, "time": 0.021242901170626282, "gflops": 1579.5597658947613, "gbs": 592.3349122105355}, {"N": 2048, "fold": 1, "time": 0.07358282720670103, "gflops": 114.00225186286492, "gbs": 684.0135111771896}, {"N": 2048, "fold": 2, "time": 0.07370002781972289, "gflops": 227.64192221254828, "gbs": 682.9257666376449}, {"N": 2048, "fold": 4, "time": 0.07357445200905204, "gflops": 456.0609163065424, "gbs": 684.0913744598137}, {"N": 2048, "fold": 8, "time": 0.07350112521089613, "gflops": 913.0317911112943, "gbs": 684.7738433334707}, {"N": 2048, "fold": 16, "time": 0.0736398187931627, "gflops": 1822.6243654535153, "gbs": 683.4841370450682}, {"N": 4096, "fold": 1, "time": 0.2832261357922107, "gflops": 118.47223034747495, "gbs": 710.8333820848497}, {"N": 4096, "fold": 2, "time": 0.2831576330121607, "gflops": 237.00178337455554, "gbs": 711.0053501236667}, {"N": 4096, "fold": 4, "time": 0.2836653312202543, "gflops": 473.1552051941996, "gbs": 709.7328077912994}, {"N": 4096, "fold": 8, "time": 0.28355305101722483, "gflops": 946.6851265997965, "gbs": 710.0138449498473}, {"N": 4096, "fold": 16, "time": 0.28309709001332517, "gflops": 1896.419747637568, "gbs": 711.157405364088}]}'

def run_benchmarks():
    results = benchmark_cuda()
    taichi_results = benchmark_taichi(max_nesting=16)
    results['taichi'] = taichi_results
    return results

def extract_gflops(result_dict):
    x = []
    y = []
    for res in result_dict:
        fma_depth = res.get("fold")
        if fma_depth == 1 or fma_depth == None:
            x.append(res["N"])
            y.append(res["gflops"])
    return x, y

def extract_bw(result_dict):
    x = []
    y = []
    for res in result_dict:
        fma_depth = res.get("fold")
        if fma_depth == 1 or fma_depth == None:
            x.append(res["N"])
            y.append(res["gbs"])
    return x, y

def extract_nested(result_dict, N=4096):
    x = []
    y = []
    for res in result_dict:
        fma_depth = res.get("fold")
        if res["N"] == N:
            if fma_depth != None:
                x.append(fma_depth)
            y.append(res["gflops"])
    return x, y


def plot_compute(results):
    print(results)
    plt.figure()
    bar_width = 0.4

    dims, y = extract_gflops(results['taichi'])
    bar_pos = [i*4 for i in range(len(y))]
    labels = ["{}x{}".format(i, i) for i in dims]
    plt.bar(bar_pos, y)

    _, y = extract_gflops(results['thrust'])
    bar_pos = [i*4+1 for i in range(len(y))]
    plt.bar(bar_pos, y)
    plt.xticks(bar_pos, labels)

    _, y = extract_gflops(results['cublas'])
    bar_pos = [i*4+2 for i in range(len(y))]
    plt.bar(bar_pos, y)
    plt.legend(['Taichi', 'CUDA/Thrust', 'CUDA/cuBLAS'])
    plt.xlabel("Array dimensions")
    plt.ylabel("Performance (GFLOPS)")
    plt.axvline(x = 7, color='grey', linestyle = 'dashed')
    plt.text(7, 120, 'L2 Cache=4MB', rotation=270)
    plt.title("Saxpy compute benchmark on 2D arrays")
    plt.savefig("fig/compute_bench.png", dpi=200)

def plot_memory_bw(results):
    plt.figure()
    dims, y = extract_bw(results['taichi'])
    bar_pos = [i*4 for i in range(len(y))]
    labels = ["{}x{}".format(i, i) for i in dims]
    plt.bar(bar_pos, y)

    _, y = extract_bw(results['thrust'])
    bar_pos = [i*4+1 for i in range(len(y))]
    plt.bar(bar_pos, y)
    plt.xticks(bar_pos, labels)

    _, y = extract_bw(results['cublas'])
    bar_pos = [i*4+2 for i in range(len(y))]
    plt.bar(bar_pos, y)
    plt.legend(['Taichi', 'CUDA/Thrust', 'CUDA/cuBLAS'])
    plt.xlabel("Array dimensions")
    plt.ylabel("Performance (GB/s)")
    plt.title("Saxpy bandwith benchmark on 2D arrays")
    plt.axhline(y = 760, color='grey', linestyle = 'dotted')
    plt.text(11, 770, 'DRAM Bandwidth=760GB/s')
    plt.axvline(x = 7, color='grey', linestyle = 'dashed')
    plt.text(7, 750, 'L2 Cache=4MB', rotation=270)
    plt.savefig("fig/bw_bench.png", dpi=200)

def plot_nested(results, N=4096):
    plt.figure()
    bar_width = 0.4
    _, y = extract_nested(results['taichi'], N)
    bar_pos = [i*4 for i in range(len(y))]
    plt.bar(bar_pos, y)

    nesting_factors, y = extract_nested(results['thrust'], N)
    bar_pos = [i*4+1 for i in range(len(y))]
    plt.bar(bar_pos, y)
    print(nesting_factors)
    labels = ["{}".format(i) for i in nesting_factors]
    plt.xticks(bar_pos, labels)

    bar_count = len(y)
    _, y = extract_nested(results['cublas'], N)
    assert(len(y) == 1)
    y = [y[0] for i in range(bar_count)]
    bar_pos = [i*4+2 for i in range(len(y))]
    plt.bar(bar_pos, y)
    plt.legend(['Taichi', 'CUDA/Thrust', 'CUDA/cuBLAS'])
    plt.xlabel("Nesting factor")
    plt.ylabel("Performance (GFLOPS)")
    plt.title("Nested saxpy compute benchmark on {}x{} arrays".format(N, N))
    plt.savefig("fig/nesting_bench_{}.png".format(N), dpi=200)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass

    if len(sys.argv) >= 2 and sys.argv[1] == 'sample':
        import json
        results = json.loads(sample_res)
    else:
        results = run_benchmarks()
    plot_compute(results)
    plot_memory_bw(results)
    plot_nested(results, 4096)
    plot_nested(results, 512)
