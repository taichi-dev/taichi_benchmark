from .nbody import run_nbody as run_baseline
from .nbody_cache_block import run_nbody as run_cache_block

def benchmark(nIters = 21):
    nBodies = 128
    baseline_results = []
    cache_block_results = []
    unroll_results = []
    for k in range(13):
        print("Benchmark nBodies={}".format(nBodies))
        print("Baseline running...")
        baseline_results.append(run_baseline(nBodies, nIters=nIters))
        print("Done.")
        print("Block running...")
        cache_block_results.append(run_cache_block(nBodies, nIters=nIters))
        print("Done.")
        nBodies *= 2
    return {"taichi_baseline": baseline_results,
            "taichi_cache_block": cache_block_results}

if __name__ == "__main__":
    print(benchmark())
