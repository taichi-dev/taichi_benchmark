# 3x3 SVD Benchmark

## Introduction 
In this benchmark, we compare performance of SVD solvers between Taichi built-in and CUDA implementations.
The Taichi built-in SVD solver implements the method described in [1], which also attaches a CUDA benchmark code repository  [3x3_SVD_CUDA](https://github.com/kuiwuchn/3x3_SVD_CUDA). We slightly modified the CUDA benchmark code in order to fit our benchmark suite, but kept all the kernel code unchanged in order to conduct a fair comparison.


## Evaluation
We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 5MB|

## Reference

[1] Gao, M., Wang, X., Wu, K., Pradhana, A., Sifakis, E., Yuksel, C., & Jiang, C. (2018). GPU optimization of material point methods. ACM Transactions on Graphics (TOG), 37(6), 1-12.