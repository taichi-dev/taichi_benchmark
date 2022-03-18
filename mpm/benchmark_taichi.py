from src.taichi.mpm3d import run_mpm3d as run_baseline

def benchmark(nIters = 2048):
    n_grids = [4, 8, 16, 32, 64]
    dt = 2e-4
    baseline_results = []
    for n_grid in n_grids:
        # adjust dt
        if (n_grid < 32):
            dt = 4e-4
        elif (n_grid < 64):
            dt = 2e-4
        else:
            dt = 8e-5
        print("Baseline running...")
        baseline_results.append(run_baseline(n_grid, dt,  nIters))
        print("Done.")
    return {"taichi_baseline": baseline_results}
if __name__ == "__main__":
    print(benchmark())
