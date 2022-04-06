from poisson_disk import run_poisson

{'numba_gpu': [{'desired_samples': 1000, 'time_ms': 17.34950498212129}, {'desired_samples': 5000, 'time_ms': 107.88116841576993}, {'desired_samples': 10000, 'time_ms': 226.49038201197982}, {'desired_samples': 50000, 'time_ms': 1260.7922151917592}, {'desired_samples': 100000, 'time_ms': 1344.771138811484}]}

def benchmark():
    n_samples = [1000, 5000, 10000, 50000, 100000]
    results = []
    for n_sample in n_samples:
        print("Numba running", "n_sample", n_sample)
        results.append(run_poisson(n_sample))
        print("Done.")
    return {"numba_gpu": results}


if __name__ == "__main__":
    print(benchmark())
