from matplotlib import pyplot as plt
from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda
import sys
import os

sample_res = {'cuda_kernel': [{'N': 2097152, 'time': 0.013, 'bandwidth': 632.447}, {'N': 4194304, 'time': 0.025, 'bandwidth': 678.898}, {'N': 8388608, 'time': 0.048, 'bandwidth': 701.305}, {'N': 16777216, 'time': 0.094, 'bandwidth': 712.983}, {'N': 33554432, 'time': 0.187, 'bandwidth': 718.707}, {'N': 67108864, 'time': 0.372, 'bandwidth': 721.921}, {'N': 134217728, 'time': 0.742, 'bandwidth': 723.445}, {'N': 268435456, 'time': 1.483, 'bandwidth': 724.217}], 'cuda_memset': [{'N': 2097152, 'time': 0.013, 'bandwidth': 639.007}, {'N': 4194304, 'time': 0.025, 'bandwidth': 677.286}, {'N': 8388608, 'time': 0.048, 'bandwidth': 699.962}, {'N': 16777216, 'time': 0.094, 'bandwidth': 712.255}, {'N': 33554432, 'time': 0.187, 'bandwidth': 718.809}, {'N': 67108864, 'time': 0.372, 'bandwidth': 721.826}, {'N': 134217728, 'time': 0.742, 'bandwidth': 723.365}, {'N': 268435456, 'time': 1.483, 'bandwidth': 724.031}], 'taichi_fill': [{'N': 2097152, 'time': 0.013420191397890448, 'bandwidth': 625.0736484517371}, {'N': 4194304, 'time': 0.024879930317401888, 'bandwidth': 674.3272905497421}, {'N': 8388608, 'time': 0.05017550475895405, 'bandwidth': 668.7412944064515}, {'N': 16777216, 'time': 0.09498817566782236, 'bandwidth': 706.4970300585887}, {'N': 33554432, 'time': 0.18780816067010164, 'bandwidth': 714.653333066623}, {'N': 67108864, 'time': 0.3733440497890115, 'bandwidth': 719.0029040283389}, {'N': 134217728, 'time': 0.7452589934691787, 'bandwidth': 720.3816615494532}, {'N': 268435456, 'time': 1.4891986697912216, 'bandwidth': 721.0198651000228}]}

def run_benchmarks():
    cuda_results = benchmark_cuda()
    taichi_results = benchmark_taichi()
    results = {**cuda_results, **taichi_results}
    return results

def get_benchmark_record(results, key):
    result_dict = results[key]
    N_arr = [rec['N'] for rec in result_dict]
    kernel_time_arr = [rec['time'] for rec in result_dict]
    bandwidth_arr = [rec['bandwidth'] for rec in result_dict]
    return N_arr, bandwidth_arr

def get_color(key):
    if key.startswith("taichi"):
        pass


def create_bar_plot(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_box_aspect(0.75)
    record_keys = ["taichi_fill",  "cuda_kernel", "cuda_memset"]
    num_records = len(record_keys)

    labels = None
    id = 0
    for key in record_keys:
        # Don't check N
        N, time = get_benchmark_record(results, key)
        bar_pos = [i*(num_records+1) + id  for i in range(len(time))]
        ax.bar(bar_pos, time)
        id += 1
        labels = ['{}MB'.format(num // 1024 // 1024 * 4) for num in N]

    bar_pos = [i*(num_records+1) + (num_records - 1) / 2.0  for i in range(len(time))]
    ax.set_xticks(bar_pos, labels, rotation=0, fontsize=10)

    title_font_size = 16
    label_font_size = 14
    ax.legend(['Taichi', 'CUDA (Kernel)', 'CUDA (cuMemset)'])
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=label_font_size)
    ax.set_xlabel("Array Size", fontsize=label_font_size)
    ax.set_title("Array Fill Benchmark", fontsize=title_font_size)
    ax.set_ylim([0, 850])
    ax.axhline(y = 760, color='grey', linestyle = 'dashed')
    ax.text(ax.get_xlim()[1] * 0.6, 770, 'DRAM Bandwidth=760GB/s')
    ax.grid("minor", axis="y")
    plt.savefig("fig/bench_svd.png", dpi=150)



if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass

    if len(sys.argv) >= 2 and sys.argv[1] == 'sample':
        results = sample_res
    else:
        results = run_benchmarks()
    print(results)
    create_bar_plot(results)
