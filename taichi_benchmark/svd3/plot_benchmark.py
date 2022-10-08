from matplotlib import pyplot as plt
from src.taichi.benchmark import benchmark as benchmark_taichi
from src.cuda.benchmark import benchmark as benchmark_cuda
import sys
import os

#sample_res = {'cuda_svd_aos': [{'N': 8192, 'kernel_time': 0.015437}, {'N': 16384, 'kernel_time': 0.017178}, {'N': 32768, 'kernel_time': 0.026621}, {'N': 65536, 'kernel_time': 0.047734}, {'N': 131072, 'kernel_time': 0.095146}, {'N': 262144, 'kernel_time': 0.179325}, {'N': 524288, 'kernel_time': 0.335946}, {'N': 1048576, 'kernel_time': 0.656682}, {'N': 2097152, 'kernel_time': 1.299488}, {'N': 4194304, 'kernel_time': 2.580931}], 'cuda_svd_soa': [{'N': 8192, 'kernel_time': 0.007229}, {'N': 16384, 'kernel_time': 0.007338}, {'N': 32768, 'kernel_time': 0.008173}, {'N': 65536, 'kernel_time': 0.015587}, {'N': 131072, 'kernel_time': 0.030678}, {'N': 262144, 'kernel_time': 0.054845}, {'N': 524288, 'kernel_time': 0.101312}, {'N': 1048576, 'kernel_time': 0.19983}, {'N': 2097152, 'kernel_time': 0.388746}, {'N': 4194304, 'kernel_time': 0.758013}], 'taichi_svd_soa': [{'N': 8192, 'kernel_time': 0.004684799956157804, 'wall_time': 0.022738613188266754}, {'N': 16384, 'kernel_time': 0.006019200012087822, 'wall_time': 0.026566162705421448}, {'N': 32768, 'kernel_time': 0.007059200061485171, 'wall_time': 0.023869797587394714}, {'N': 65536, 'kernel_time': 0.014515200071036816, 'wall_time': 0.029187602922320366}, {'N': 131072, 'kernel_time': 0.02732799984514713, 'wall_time': 0.0417312141507864}, {'N': 262144, 'kernel_time': 0.05061759911477566, 'wall_time': 0.06944923661649227}, {'N': 524288, 'kernel_time': 0.10014719665050506, 'wall_time': 0.12033055536448956}, {'N': 1048576, 'kernel_time': 0.19497600346803665, 'wall_time': 0.22520381025969982}, {'N': 2097152, 'kernel_time': 0.4029279977083206, 'wall_time': 0.45370832085609436}, {'N': 4194304, 'kernel_time': 0.8099840104579925, 'wall_time': 0.9023827966302633}], 'taichi_svd_aos': [{'N': 8192, 'kernel_time': 0.009065600112080574, 'wall_time': 0.019104918465018272}, {'N': 16384, 'kernel_time': 0.014438400138169527, 'wall_time': 0.02398798242211342}, {'N': 32768, 'kernel_time': 0.0243840005248785, 'wall_time': 0.0334025826305151}, {'N': 65536, 'kernel_time': 0.04248959943652153, 'wall_time': 0.05279681645333767}, {'N': 131072, 'kernel_time': 0.07956479713320733, 'wall_time': 0.09165341034531593}, {'N': 262144, 'kernel_time': 0.15032320320606232, 'wall_time': 0.17172172665596008}, {'N': 524288, 'kernel_time': 0.3009728044271469, 'wall_time': 0.3374490886926651}, {'N': 1048576, 'kernel_time': 0.5963776111602783, 'wall_time': 0.6607457064092159}, {'N': 2097152, 'kernel_time': 1.1851104140281676, 'wall_time': 1.309581520035863}, {'N': 4194304, 'kernel_time': 2.3683903932571413, 'wall_time': 2.6098044123500586}]}

def run_benchmarks():
    cuda_results = benchmark_cuda()
    taichi_results = benchmark_taichi()
    results = {**cuda_results, **taichi_results}
    return results

def get_benchmark_record(results, key):
    result_dict = results[key]
    N_arr = [rec['N'] for rec in result_dict]
    kernel_time_arr = [rec['kernel_time'] for rec in result_dict]
    return N_arr, kernel_time_arr

def get_color(key):
    if key.startswith("taichi"):
        pass


def create_bar_plot(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_box_aspect(0.75)
    record_keys = ["taichi_svd_aos",  "cuda_svd_aos", "taichi_svd_soa", "cuda_svd_soa"]
    num_records = len(record_keys)

    labels = None
    id = 0
    for key in record_keys:
        # Don't check N
        N, time = get_benchmark_record(results, key)
        bar_pos = [i*(num_records+1) + id  for i in range(len(time))]
        ax.bar(bar_pos, time)
        id += 1
        labels = [str(num) for num in N]

    bar_pos = [i*(num_records+1) + (num_records - 1) / 2.0  for i in range(len(time))]
    ax.set_xticks(bar_pos, labels, rotation=20, fontsize=8)

    title_font_size = 16
    label_font_size = 14
    ax.legend(['Taichi (AOS)', 'CUDA (AOS)', 'Taichi (SOA)', 'CUDA (SOA)'])
    ax.set_ylabel("Kernel Elapsed Time (ms)", fontsize=label_font_size)
    ax.set_xlabel("#Tiles (3x3 matrices)", fontsize=label_font_size)
    ax.set_title("3x3 SVD Benchmark", fontsize=title_font_size)
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
