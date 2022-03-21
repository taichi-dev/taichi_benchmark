import numpy as np
from path_tracing import run_smallpt as run_pt

def benchmark():
    spps_gpu = np.arange(16, 256+16, 16).tolist()
    results = []
    for spp in spps_gpu:
        print("SmallPT running", "spp", spp)
        results.append(run_pt(spp))
        print("Done.")
    return {"taichi_baseline": results}

if __name__ == "__main__":
    print(benchmark())
