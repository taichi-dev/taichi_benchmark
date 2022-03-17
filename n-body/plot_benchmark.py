from benchmark_cuda import benchmark as benchmark_cuda
from benchmark_taichi import benchmark as benchmark_taichi

import matplotlib.pyplot as plt
import sys
import os


cuda_sample_results = {'cuda_baseline': [{'nbodies': 128, 'rate': 1.053, 'time': 0.016}, {'nbodies': 256, 'rate': 2.706, 'time': 0.024}, {'nbodies': 512, 'rate': 6.258, 'time': 0.042}, {'nbodies': 1024, 'rate': 13.54, 'time': 0.077}, {'nbodies': 2048, 'rate': 26.753, 'time': 0.157}, {'nbodies': 4096, 'rate': 53.525, 'time': 0.313}, {'nbodies': 8192, 'rate': 108.26, 'time': 0.62}, {'nbodies': 16384, 'rate': 217.161, 'time': 1.236}, {'nbodies': 32768, 'rate': 408.75, 'time': 2.627}, {'nbodies': 65536, 'rate': 433.684, 'time': 9.903}, {'nbodies': 131072, 'rate': 487.461, 'time': 35.244}, {'nbodies': 262144, 'rate': 501.46, 'time': 137.039}, {'nbodies': 524288, 'rate': 516.159, 'time': 532.545}], 'cuda_block': [{'nbodies': 128, 'rate': 0.702, 'time': 0.023}, {'nbodies': 256, 'rate': 2.756, 'time': 0.024}, {'nbodies': 512, 'rate': 6.741, 'time': 0.039}, {'nbodies': 1024, 'rate': 16.852, 'time': 0.062}, {'nbodies': 2048, 'rate': 41.346, 'time': 0.101}, {'nbodies': 4096, 'rate': 89.931, 'time': 0.187}, {'nbodies': 8192, 'rate': 188.626, 'time': 0.356}, {'nbodies': 16384, 'rate': 383.358, 'time': 0.7}, {'nbodies': 32768, 'rate': 503.841, 'time': 2.131}, {'nbodies': 65536, 'rate': 532.126, 'time': 8.071}, {'nbodies': 131072, 'rate': 536.981, 'time': 31.993}, {'nbodies': 262144, 'rate': 545.344, 'time': 126.011}, {'nbodies': 524288, 'rate': 561.48, 'time': 489.56}], 'cuda_best': [{'nbodies': 128, 'rate': 1.076, 'time': 0.015}, {'nbodies': 256, 'rate': 4.068, 'time': 0.016}, {'nbodies': 512, 'rate': 10.082, 'time': 0.026}, {'nbodies': 1024, 'rate': 27.354, 'time': 0.038}, {'nbodies': 2048, 'rate': 60.983, 'time': 0.069}, {'nbodies': 4096, 'rate': 129.276, 'time': 0.13}, {'nbodies': 8192, 'rate': 269.273, 'time': 0.249}, {'nbodies': 16384, 'rate': 553.983, 'time': 0.485}, {'nbodies': 32768, 'rate': 691.844, 'time': 1.552}, {'nbodies': 65536, 'rate': 753.474, 'time': 5.7}, {'nbodies': 131072, 'rate': 758.679, 'time': 22.644}, {'nbodies': 262144, 'rate': 763.257, 'time': 90.035}, {'nbodies': 524288, 'rate': 785.431, 'time': 349.971}]}

taichi_sample_results = {'taichi_baseline': [{'nbodies': 128, 'time': 0.017808408153300384, 'rate': 0.9200148524765026}, {'nbodies': 256, 'time': 0.022995228670081313, 'rate': 2.8499825307355056}, {'nbodies': 512, 'time': 0.04559633683185188, 'rate': 5.74923378092242}, {'nbodies': 1024, 'time': 0.09497817681760204, 'rate': 11.040178229718032}, {'nbodies': 2048, 'time': 0.19198047871492346, 'rate': 21.847554647617194}, {'nbodies': 4096, 'time': 0.3837176731654576, 'rate': 43.72281282120078}, {'nbodies': 8192, 'time': 0.7712500435965401, 'rate': 87.0131088577368}, {'nbodies': 16384, 'time': 1.5502462581712373, 'rate': 173.15665468314847}, {'nbodies': 32768, 'time': 3.1822457605478713, 'rate': 337.41637346549226}, {'nbodies': 65536, 'time': 10.529143469674247, 'rate': 407.9123157900021}, {'nbodies': 131072, 'time': 39.45852785694356, 'rate': 435.3905256244079}, {'nbodies': 262144, 'time': 135.64622158906897, 'rate': 506.60811581011814}, {'nbodies': 524288, 'time': 542.3727570747842, 'rate': 506.80625705929197}], 'taichi_block': [{'nbodies': 128, 'time': 0.017171003380600288, 'rate': 0.9541667214689713}, {'nbodies': 256, 'time': 0.020212056685467154, 'rate': 3.242421145945113}, {'nbodies': 512, 'time': 0.028975155888771524, 'rate': 9.047198952313014}, {'nbodies': 1024, 'time': 0.04869577836017219, 'rate': 21.53320134333493}, {'nbodies': 2048, 'time': 0.07942258095254703, 'rate': 52.809968521496295}, {'nbodies': 4096, 'time': 0.13922671882473692, 'rate': 120.50284702263005}, {'nbodies': 8192, 'time': 0.29348840518873565, 'rate': 228.6593364969353}, {'nbodies': 16384, 'time': 0.708687062166175, 'rate': 378.7785474444804}, {'nbodies': 32768, 'time': 2.3187325925243143, 'rate': 463.0727266532528}, {'nbodies': 65536, 'time': 8.784848816540777, 'rate': 488.9062277216554}, {'nbodies': 131072, 'time': 35.021874369407186, 'rate': 490.5468223313372}, {'nbodies': 262144, 'time': 137.99058661168937, 'rate': 498.00119285947494}, {'nbodies': 524288, 'time': 545.8203286540752, 'rate': 503.6051105348432}], 'taichi_unroll': [{'nbodies': 128, 'time': 0.020689434475368924, 'rate': 0.7919017805685018}, {'nbodies': 256, 'time': 0.02948443094889323, 'rate': 2.2227324011644205}, {'nbodies': 512, 'time': 0.030491087171766493, 'rate': 8.597397610759339}, {'nbodies': 1024, 'time': 0.04863739013671875, 'rate': 21.55905152501961}, {'nbodies': 2048, 'time': 0.07565816243489583, 'rate': 55.43756106433614}, {'nbodies': 4096, 'time': 0.13568666246202257, 'rate': 123.64675860971808}, {'nbodies': 8192, 'time': 0.2568562825520833, 'rate': 261.27008976855444}, {'nbodies': 16384, 'time': 0.621742672390408, 'rate': 431.7468752272524}, {'nbodies': 32768, 'time': 2.151118384467231, 'rate': 499.15515192155937}, {'nbodies': 65536, 'time': 8.204380671183268, 'rate': 523.4968327451538}, {'nbodies': 131072, 'time': 32.951858308580185, 'rate': 521.3626807665233}, {'nbodies': 262144, 'time': 124.52809015909831, 'rate': 551.8391605315983}, {'nbodies': 524288, 'time': 508.0065197414822, 'rate': 541.0912975760266}]}

def run_benchmarks():
    return benchmark_cuda(), benchmark_taichi()

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["rate"])
    return perf

def extract_nbodies(results):
    nbodies = []
    for record in results:
        nbodies.append(record["nbodies"])
    return nbodies

def plot(cuda_results, taichi_results, plot_cuda_roofline=True):
    plt.figure()
    x = extract_nbodies(taichi_results["taichi_baseline"])
    plt.plot(x, extract_perf(taichi_results["taichi_baseline"]), marker='o')
    plt.plot(x, extract_perf(taichi_results["taichi_block"]), marker='o')
    plt.plot(x, extract_perf(taichi_results["taichi_unroll"]), marker='o')
    plt.plot(x, extract_perf(cuda_results["cuda_baseline"]), marker='P', ms=6)
    plt.plot(x, extract_perf(cuda_results["cuda_block"]), marker='P', ms=6)
    if plot_cuda_roofline:
        plt.plot(x, extract_perf(cuda_results["cuda_best"]), marker='P', ms=6)
    plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("#Bodies")
    plt.ylabel("Speed (billion body interactions per second)")
    plt.legend(["Taichi/Baseline", "Taichi/Block", "Taichi/Unroll", "CUDA/Baseline", "CUDA/Block", "CUDA/Roofline"], loc='lower right')
    plt.title("N-Body benchmark")
    if plot_cuda_roofline:
        plt.savefig("fig/bench_roofline.png", dpi=150)
    else:
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
    plot(cuda_results, taichi_results, plot_cuda_roofline=True)
    plot(cuda_results, taichi_results, plot_cuda_roofline=False)
