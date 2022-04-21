# Array Sum Benchmark

## Introduction

Summing up all elements in an array is a frequent operation in data analysis, deep learning, and physical simulations. The operation accepts an `N`-element array and outputs the sum of all elements. For parallel hardware like GPUs, a widely adopted optimization technique is known as reduction. You can refer to [Nvidia's official document](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) for details.

In this benchmark, we compare the Taichi implementation against CUDA programs. The Taichi version of sum is simply implemented with `atomic_add`, and relies on the compiler to automatically apply the parallelization and reduction optimization. 
```python
@ti.kernel 
def reduce_sum_kernel():
    sum = 0.0
    for i in ti.grouped(f):
        ti.atomic_add(sum, f[i])
```
For `CUDA` implementations, we take three different approaches: 1) homegrown CUDA implementation 2) Nvidia CUB and 3) Nvidia thrust. We have validated that all four approaches generate identical results.

## Evaluation

We conduct performance evaluations on two GPU devices, the RTX 3080 and RTX 2060, respectively.

| Device            | Nvidia RTX 3080 (10GB) | Nvidia RTX 2060 (6GB) |
| ----------------- | ---------------------- | -------------------- |
| FP32 performance  | 29700 GFLOPS           | 6451 GFLOPS          |
| Memory bandwidth  | 760 GB/s               | 336 GB/s             |
| L2 cache capacity | 5 MB                   | 3 MB                 |
| Driver version    | 470.57.02              | 510.47.03            |
| CUDA version      | 11.4                   | 11.6                 |

Performance is measured in memory access bandwidth(GB/s), higher is better. Data type is `fp32`. The results are averaged from 5 consecutive launches following a warm-up run. The warm-up overhead is excluded from timers.

- Nvidia RTX 3080
<p align='center'>
<img src="./fig/compute_bench_3080.png" alt="3080" width=600 /></p>


- Nvidia RTX 2060
<p align='center'>
<img src="./fig/compute_bench_2060.png" alt="2060" width=600 /></p>

As is illustrated in the figures, Taichi can leverage up to 89.6% and 62.3% of peak bandwidth on RTX2080 and RTX3080, while the CUDA top performer, CUB, achieves 90.6% and 84.8%, respectively. In all cases, we find that performance of Taichi and CUB are comparable, slighly better than our manually implemented CUDA kernels. The thrust library which adopts CUB as its underlying compute engine, is the bottom in this benchmark. 

## Reproduction Steps

- Pre-requisites

```shell
python3 -m pip install --upgrade taichi
python3 -m pip install matplotlib
```

If you want to compare with `CUDA`, make sure you have `nvcc` properly installed.

- Run the benchmark and draw the plots

```shell
python3 plot_benchmark.py
```

