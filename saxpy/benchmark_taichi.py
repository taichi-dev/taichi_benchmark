from src.taichi.saxpy import saxpy as taichi_saxpy

def benchmark_taichi(max_fma=512, print_res=True):
    rd_arr = []
    for i in [256, 512, 1024, 2048, 4096]:
        for j in range(12):
            if 2**j > max_fma:
                break
            rd = taichi_saxpy(i, 2**(j))
            if print_res:
                print("{}x{}@{}, {:.3f}ms, {:.3f} GFLOPS, {:.3f} GB/s".format(rd["N"], rd["N"], rd["fold"], rd["time"], rd["gflops"], rd["gbs"]))
            rd_arr.append(rd)
    return rd_arr

if __name__ == '__main__':
    benchmark_taichi()
