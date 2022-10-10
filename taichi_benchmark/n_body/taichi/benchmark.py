from .nbody import nbody as run_baseline
from .nbody_cache_block import nbody as run_cache_block

def benchmark(nIters = 21):
    nBodies = 128
    results = []
    # Full test configuration
    # for k in range(13):
    for k in range(3):
        print("Benchmark nBodies={}".format(nBodies))
        print("Baseline running...")
        results += run_baseline(nBodies=nBodies)
        print("Done.")
        print("Block running...")
        results += run_cache_block(nBodies=nBodies)
        print("Done.")
        nBodies *= 2
    return results

if __name__ == "__main__":
    print(benchmark())
