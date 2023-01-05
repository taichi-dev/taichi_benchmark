# Stencil Benchmark

## Introduction

Stencil computation is widely employed in engineering simulations. In this benchmark, we employ the 5-point [iterative stencil algorithm](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops). As the algorithm is simple, we develop our own CUDA and Taichi implementations for benchmark, respectively. 

## Evaluation

We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 5 MB|
|Driver version| 470.57.02|
|CUDA version| 11.4|


Perfomrmance is measured with the achieved memory access bandwidth, as stencil computation appears to be memory-bound.
The bandwidth is calculated by `1e-9 * 2 * N * N * sizeof(float) / t`, where `t` is the compute time measured in seconds, `NxN` stands for matrix shape.
The performance data of different Taichi and CUDA implementations are illustrated as follows.

<p align="center">
<img src="fig/bench.png" width="600">
</p>

In this Figure, we notice that CUDA and Taichi implementations reveal similar performance. For small matrices, Taichi is slightly bothered by kernel launch overhead from Python side. For large matrices, the two implementations are neck-to-neck and approach the roofline of GPU bandwidth. Therefore, we can conclude that both implementations are high-performance that can fully leverage GPU's capability.


## Reproduction Steps

* Pre-requisites
```shell
python3 -m pip install --upgrade taichi
python3 -m pip install matplotlib
```
If you want to compare with CUDA, make sure you have `nvcc` properly installed.

* Run the benchmark and draw the plots
```shell
cd stencil2d
python3 plot_benchmark.py
```
