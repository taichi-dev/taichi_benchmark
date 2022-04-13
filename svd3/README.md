# 3x3 SVD Benchmark

## Introduction 
In this benchmark, we compare performance of SVD solvers between Taichi built-in and CUDA implementations.
The Taichi built-in SVD solver implements the method described in [1], which also attaches a carefully optimized CUDA code repository [3x3_SVD_CUDA](https://github.com/kuiwuchn/3x3_SVD_CUDA). 
We slightly modified the CUDA benchmark code in order to fit our benchmark suite, but kept all the SVD kernel code unchanged in order to conduct a fair comparison.


## Evaluation
We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 5 MB|
|Driver version| 470.57.02|
|CUDA version| 11.4|

Performance is measured as the kernel compute time measured with the `cudaEvent` APIs, lower is better. The unit is milliseconds (ms). In each experiment, we first conduct a warm-up run, and time for 10 repeated invokes.

<p align="center">
<img src="fig/bench_svd.png" width="600">
</p>

The figure reveals that Taichi slightly outperform the CUDA implementation in the AOS layout. The performance in SOA layout is neck-to-neck. We have also noticed that the overall performance is generally bound by memory access efficiency. The results indicate that the compute kernels implemented with Taichi and CUDA are both highly efficient.

## Reproduction Steps

* Pre-requisites
```shell
python3 -m pip install --upgrade taichi
python3 -m pip install matplotlib
```
If you want to compare with CUDA, make sure you have `nvcc` properly installed.

* Run the benchmark and draw the plots
```shell
python3 plot_benchmark.py
```

## Reference

[1] Gao, M., Wang, X., Wu, K., Pradhana, A., Sifakis, E., Yuksel, C., & Jiang, C. (2018). GPU optimization of material point methods. ACM Transactions on Graphics (TOG), 37(6), 1-12.
