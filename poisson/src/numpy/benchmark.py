from .poisson_disk import run_poisson

{'numpy_cpu': [{'desired_samples': 1000, 'time_ms': 991.2736028200015}, {'desired_samples': 5000, 'time_ms': 5998.027631384321}, {'desired_samples': 10000, 'time_ms': 12623.283750191331}, {'desired_samples': 50000, 'time_ms': 70075.99696719553}, {'desired_samples': 100000, 'time_ms': 74836.31602639798}]}

def benchmark():
    n_samples = [1000, 5000, 10000, 50000, 100000]
    results = []
    for n_sample in n_samples:
        print("Numpy running", "n_sample", n_sample)
        results.append(run_poisson(n_sample))
        print("Done.")
    return {"numpy_cpu": results}


if __name__ == "__main__":
    print(benchmark())
