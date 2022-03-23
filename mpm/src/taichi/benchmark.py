import numpy as np
from mpm2d import run_mpm as run_mpm_2d
from mpm3d import run_mpm as run_mpm_3d

def benchmark(nIters = 2048):
    n_grids = np.arange(32, 256+32, 32).tolist()
    mpm_2d_results = []
    mpm_3d_results = []
    for n_grid in n_grids:
        print("MPM 2D running", "n_grid", n_grid)
        mpm_2d_results.append(run_mpm_2d(n_grid))
        print("Done.")
        print("MPM 3D running", "n_grid", n_grid)
        mpm_3d_results.append(run_mpm_3d(n_grid))
        print("Done.")
    return {"taichi_2d": mpm_2d_results, "taichi_3d": mpm_3d_results}
if __name__ == "__main__":
    print(benchmark())
