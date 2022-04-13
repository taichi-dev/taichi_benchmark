# Particle to Grid (P2G) Benchmark

## Introduction
In this benchmark, we compare a simple implementation of the P2G method in Taichi
to its ported version in CUDA.
The measured CUDA implementation is based on the p2g kernel from a 
[MPM3D open-source implementation](https://github.com/Aisk1436/mpm3d) 
 that was originally written in 
[Taichi](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm3d.py). 

## Evaluation
We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 5 MB|

Performance is measured in milliseconds per frame (ms), and plotted against
different number of particles.

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
python3 plot_benchmark.py
```
