import matplotlib.pyplot as plt
import sys
import os

cuda_sample_results = {'cuda_2d': [{'n_particles': 128, 'fps': 995}, {'n_particles': 512, 'fps': 970}, {'n_particles': 1152, 'fps': 961}, {'n_particles': 2048, 'fps': 963}, {'n_particles': 3200, 'fps': 943}, {'n_particles': 4608, 'fps': 877}, {'n_particles': 6272, 'fps': 863}, {'n_particles': 8192, 'fps': 803}, {'n_particles': 10368, 'fps': 752}, {'n_particles': 12800, 'fps': 703}, {'n_particles': 15488, 'fps': 620}, {'n_particles': 18432, 'fps': 560}, {'n_particles': 21632, 'fps': 547}, {'n_particles': 25088, 'fps': 475}, {'n_particles': 28800, 'fps': 452}, {'n_particles': 32768, 'fps': 417}], 'cuda_3d': [{'n_particles': 1024, 'fps': 446}, {'n_particles': 8192, 'fps': 368}, {'n_particles': 27648, 'fps': 200}, {'n_particles': 65536, 'fps': 117}, {'n_particles': 128000, 'fps': 55}, {'n_particles': 221184, 'fps': 27}, {'n_particles': 351232, 'fps': 12}, {'n_particles': 524288, 'fps': 5}, {'n_particles': 746496, 'fps': 3}, {'n_particles': 1024000, 'fps': 1}, {'n_particles': 1362944, 'fps': 1}, {'n_particles': 1769472, 'fps': 0}, {'n_particles': 2249728, 'fps': 0}, {'n_particles': 2809856, 'fps': 0}, {'n_particles': 3456000, 'fps': 0}, {'n_particles': 4194304, 'fps': 0}]}


#cuda_sample_results = {'cuda_baseline': [{'n_particles': 128, 'fps': 997}, {'n_particles': 512, 'fps': 981}, {'n_particles': 1152, 'fps': 976}, {'n_particles': 2048, 'fps': 972}, {'n_particles': 3200, 'fps': 951}, {'n_particles': 4608, 'fps': 893}, {'n_particles': 6272, 'fps': 878}, {'n_particles': 8192, 'fps': 819}, {'n_particles': 10368, 'fps': 748}, {'n_particles': 12800, 'fps': 683}, {'n_particles': 15488, 'fps': 634}, {'n_particles': 18432, 'fps': 569}, {'n_particles': 21632, 'fps': 554}, {'n_particles': 25088, 'fps': 481}, {'n_particles': 28800, 'fps': 447}, {'n_particles': 32768, 'fps': 422}]}

#taichi_sample_results = {'taichi_baseline': [{'n_particles': 128, 'fps': 2126}, {'n_particles': 512, 'fps': 2135}, {'n_particles': 1152, 'fps': 2134}, {'n_particles': 2048, 'fps': 2116}, {'n_particles': 3200, 'fps': 2078}, {'n_particles': 4608, 'fps': 2005}, {'n_particles': 6272, 'fps': 1951}, {'n_particles': 8192, 'fps': 1800}, {'n_particles': 10368, 'fps': 1677}, {'n_particles': 12800, 'fps': 1607}, {'n_particles': 15488, 'fps': 1417}, {'n_particles': 18432, 'fps': 1138}, {'n_particles': 21632, 'fps': 1044}, {'n_particles': 25088, 'fps': 970}, {'n_particles': 28800, 'fps': 891}, {'n_particles': 32768, 'fps': 828}]}

taichi_sample_results = {'taichi_2d': [{'n_particles': 128, 'fps': 2105}, {'n_particles': 512, 'fps': 2088}, {'n_particles': 1152, 'fps': 2083}, {'n_particles': 2048, 'fps': 2083}, {'n_particles': 3200, 'fps': 2027}, {'n_particles': 4608, 'fps': 1959}, {'n_particles': 6272, 'fps': 1902}, {'n_particles': 8192, 'fps': 1755}, {'n_particles': 10368, 'fps': 1579}, {'n_particles': 12800, 'fps': 1462}, {'n_particles': 15488, 'fps': 1318}, {'n_particles': 18432, 'fps': 1178}, {'n_particles': 21632, 'fps': 1088}, {'n_particles': 25088, 'fps': 1004}, {'n_particles': 28800, 'fps': 904}, {'n_particles': 32768, 'fps': 816}], 'taichi_3d': [{'n_particles': 1024, 'fps': 1163}, {'n_particles': 8192, 'fps': 523}, {'n_particles': 27648, 'fps': 194}, {'n_particles': 65536, 'fps': 95}, {'n_particles': 128000, 'fps': 46}, {'n_particles': 221184, 'fps': 29}, {'n_particles': 351232, 'fps': 15}, {'n_particles': 524288, 'fps': 7}, {'n_particles': 746496, 'fps': 3}, {'n_particles': 1024000, 'fps': 2}, {'n_particles': 1362944, 'fps': 1}, {'n_particles': 1769472, 'fps': 1}, {'n_particles': 2249728, 'fps': 1}, {'n_particles': 2809856, 'fps': 0}, {'n_particles': 3456000, 'fps': 0}, {'n_particles': 4194304, 'fps': 0}]}

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["fps"])
    return perf

def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["n_particles"])
    return particles 

def plot(cuda_results, taichi_results):
    plt.figure()
    #x_2d = extract_particles(cuda_results["cuda_2d"])
    x_3d = extract_particles(cuda_results["cuda_3d"])
    #plt.plot(x_2d, extract_perf(cuda_results["cuda_2d"]), marker='s')
    #plt.plot(x_2d, extract_perf(taichi_results["taichi_2d"]), marker='o')
    plt.plot(x_3d, extract_perf(cuda_results["cuda_3d"]), marker='s')
    plt.plot(x_3d, extract_perf(taichi_results["taichi_3d"]), marker='o')

    plt.xscale('log')
    plt.grid('minor')
    plt.xlabel("#Particles")
    plt.ylabel("Frames per Second")
    plt.legend(["CUDA 2D", "Taichi 2D", "CUDA 3D", "Taichi 3D"], loc='upper right')
    plt.title("MPM benchmark")
    #plt.savefig("fig/bench_2d.png", dpi=150)
    plt.savefig("fig/bench_3d.png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    cuda_results = cuda_sample_results
    taichi_results = taichi_sample_results
    plot(cuda_results, taichi_results)
