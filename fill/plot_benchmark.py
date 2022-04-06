from matplotlib import pyplot as plt
from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda
import sys
import os

sample_res = {'cuda_kernel': [{'N': 2097152, 'time': 0.013, 'bandwidth': 632.938}, {'N': 4194304, 'time': 0.025, 'bandwidth': 678.134}, {'N': 8388608, 'time': 0.048, 'bandwidth': 701.28}, {'N': 16777216, 'time': 0.094, 'bandwidth': 712.921}, {'N': 33554432, 'time': 0.187, 'bandwidth': 718.698}, {'N': 67108864, 'time': 0.372, 'bandwidth': 721.962}, {'N': 134217728, 'time': 0.742, 'bandwidth': 723.445}, {'N': 268435456, 'time': 1.483, 'bandwidth': 724.197}], 'cuda_memset': [{'N': 2097152, 'time': 0.013, 'bandwidth': 639.448}, {'N': 4194304, 'time': 0.025, 'bandwidth': 676.533}, {'N': 8388608, 'time': 0.048, 'bandwidth': 700.231}, {'N': 16777216, 'time': 0.094, 'bandwidth': 712.038}, {'N': 33554432, 'time': 0.187, 'bandwidth': 718.778}, {'N': 67108864, 'time': 0.372, 'bandwidth': 721.81}, {'N': 134217728, 'time': 0.742, 'bandwidth': 723.387}, {'N': 268435456, 'time': 1.483, 'bandwidth': 724.223}], 'taichi_fill': [{'N': 2097152, 'time': 0.013391969660297036, 'bandwidth': 626.3909053549886}, {'N': 4194304, 'time': 0.024852906176820396, 'bandwidth': 675.0605293656818}, {'N': 8388608, 'time': 0.04924052860587835, 'bandwidth': 681.4393133057117}, {'N': 16777216, 'time': 0.09494293201714754, 'bandwidth': 706.8337007738453}, {'N': 33554432, 'time': 0.18779275473207235, 'bandwidth': 714.7119610204935}, {'N': 67108864, 'time': 0.3734815325587988, 'bandwidth': 718.7382309398097}, {'N': 134217728, 'time': 0.7453848794102669, 'bandwidth': 720.2599983310114}, {'N': 268435456, 'time': 1.4887062897905707, 'bandwidth': 721.2583377685954}]}

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
    ax.set_xlabel("#Elements", fontsize=label_font_size)
    ax.set_title("Fill bandwidth Benchmark", fontsize=title_font_size)
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
