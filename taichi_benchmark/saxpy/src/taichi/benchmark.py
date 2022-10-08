from .saxpy import saxpy 

def benchmark(max_nesting=512, print_res=True):
    results = []
    for i in [256, 512, 1024, 2048, 4096]:
        for j in range(20):
            if 2**j > max_nesting:
                break
            repeats = 500
            if 2 ** j < 256 and i < 2048:
                repeats = 2000
            curr_results = saxpy(N=i, len_coeff=2**j, repeats=repeats)
            results += curr_results
    if print_res:
        for res in results:
            arch = res["config"]["arch"]
            N = res["config"]["N"]
            fold = res["config"]["len_coeff"]
            gflops = res["metrics"]["GFLOPS"]
            gbs = res["metrics"]["GB/s"]
            print("Arch {} {}x{}@{}, {:.3f}ms, {:.3f} GFLOPS, {:.3f} GB/s".format(arch, N, N, fold, res["wall_time"], gflops, gbs))
    return results

if __name__ == '__main__':
    benchmark()