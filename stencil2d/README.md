# Stencil Benchmark

## Introduction

Stencil computation is widely employed in engineering simulations. In this benchmark, we employ the most simple 5-point [iterative stencil algorithm](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops).

## Implementation

## Evaluation

We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 5 MB|


The performance of different Taichi and CUDA implementations is illustrated in the figure.

<p align="center">
<img src="fig/bench.png" width="560">
</p>


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
