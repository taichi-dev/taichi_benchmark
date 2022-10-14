import numpy as np
from .p2g import run_p2g


def benchmark(nIters=2048):
    n_grids = np.arange(32, 256 + 32, 32).tolist()
    results = []
    for n_grid in n_grids:
        print("P2G is running", "n_grid", n_grid)
        results.append(run_p2g(n_grid))
        print("Done.")
    return {"taichi_baseline": results}


if __name__ == "__main__":
    print(benchmark())
