from .poisson_disk import run_poisson

{'taichi_cpu': [{'desired_samples': 1000, 'time_ms': 14.330414799042046}, {'desired_samples': 5000, 'time_ms': 69.50833240989596}, {'desired_samples': 10000, 'time_ms': 138.9469366054982}, {'desired_samples': 50000, 'time_ms': 691.8933660024777}, {'desired_samples': 100000, 'time_ms': 755.0804344005883}]}

def benchmark():
    n_samples = [1000, 5000, 10000, 50000, 100000]
    results = []
    for n_sample in n_samples:
        print("CPU running", "n_sample", n_sample)
        results.append(run_poisson(n_sample))
        print("Done.")
    return {"taichi_cpu": results}


if __name__ == "__main__":
    print(benchmark())
