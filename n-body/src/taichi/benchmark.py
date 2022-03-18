from .nbody import run_nbody as run_baseline
from .nbody_block import run_nbody as run_block
from .nbody_explicit_unroll import run_nbody as run_unroll

def benchmark(nIters = 21):
    nBodies = 128
    baseline_results = []
    block_results = []
    unroll_results = []
    for k in range(13):
        print("Benchmark nBodies={}".format(nBodies))
        print("Baseline running...")
        baseline_results.append(run_baseline(nBodies, nIters=nIters))
        print("Done.")
        print("Block running...")
        block_results.append(run_block(nBodies, nIters=nIters))
        print("Done.")
        print("Unroll running...")
        unroll_results.append(run_unroll(nBodies, nIters=nIters))
        print("Done.")
        nBodies *= 2
    return {"taichi_baseline": baseline_results,
            "taichi_block": block_results,
            "taichi_unroll": unroll_results}

if __name__ == "__main__":
    print(benchmark())
