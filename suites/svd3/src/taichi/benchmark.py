from .svd_soa import SVD as run_svd_soa
from .svd_aos import SVD as run_svd_aos
import time

def benchmark(nIters = 10):
    soa_results = []
    aos_results = []
    N = 8192
    for k in range(8):
        soa_results.append(run_svd_soa(N, nIters))
        aos_results.append(run_svd_aos(N, nIters))
        N *= 2

    return {"taichi_svd_soa": soa_results,
            "taichi_svd_aos": aos_results}

if __name__ == "__main__":
    print(benchmark())
