import os
import numpy as np

from multiprocessing import Process, Manager

def run_jax_benchmark(test_name, results):
    from .smoke_jax import run_smoke as run_smoke_jax
    n_steps = np.arange(25, 200 + 25, 25).tolist()
    jax_results = []
    if test_name.endswith('_cpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        dev_str = "CPU"
    elif test_name.endswith('_gpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        dev_str = "GPU"
    else:
        raise RuntimeError('Illegal test name {}. Test name must ends with "_cpu" or "_gpu"'.format(test_name))

    for n_step in n_steps:
        print("Jax {} running n_step {}".format(dev_str, n_step))
        jax_results.append(run_smoke_jax(n_step))
        print("Done.")
    results[test_name] = jax_results


def benchmark():
    manager = Manager()
    results = manager.dict()
    p = Process(target=run_jax_benchmark, args=('jax_gpu', results))
    p.start()
    p.join()
    p = Process(target=run_jax_benchmark, args=('jax_cpu', results))
    p.start()
    p.join()
    return results


if __name__ == "__main__":
    print(benchmark())
