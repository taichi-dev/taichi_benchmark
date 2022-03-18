import mpm

def benchmark(nIters = 2048):
    configs = [(16, 20), (32, 20), (48, 20), (64, 20), 
               (80, 20), (96, 20), (112, 20), (128, 20), 
               (144, 32), (160, 32), (176, 32), (192, 32), 
               (208, 32), (224, 32), (240, 32), (256, 32)]
    # Taichi does not allow power of two
    #configs = [(16, 20), (32, 20), (64, 20), 
    #           (128, 20), (256, 32)]
    baseline_results = []
    for n_grid, step in configs:
        print("Baseline running", "n_grid", n_grid, "step", step)
        baseline_results.append(mpm.run_mpm(n_grid, step))
        print("Done.")
    return {"taichi_baseline": baseline_results}
if __name__ == "__main__":
    print(benchmark())
