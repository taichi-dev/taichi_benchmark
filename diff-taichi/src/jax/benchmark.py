import numpy as np
from smoke_jax import run_smoke as run_smoke_jax


def benchmark():
    jax_env = "jax_gpu"
    n_steps = np.arange(25, 200 + 25, 25).tolist()
    jax_results = []
    for n_step in n_steps:
        print("Jax running", "n_step", n_step)
        jax_results.append(run_smoke_jax(n_step))
        print("Done.")
    return {jax_env: jax_results}


if __name__ == "__main__":
    print(benchmark())
