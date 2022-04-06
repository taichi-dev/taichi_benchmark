# Taichi Benchmark

<!-- Purpose -->
The Taichi programming language is knwon for attaining high performance with easily understandable programs. The elegant parallel programming style has attracted thousands of users to the Taichi community and we improve the compiler together. 
Compared with programming styles, performance cannot easily measure with a single glance at the code. Moreover, general benchmark cases are insuitable as Taichi is a domain-specific language (DSL) with its own problem set to solve. 

Therefore, we kick off the Taichi benchmark project to answer two questions regarding compute performance: 1) how fast is Taichi and 2) how much faster can Taichi progress in the future. In this repository, we exhibit Taichi's performance on a set of benchmarks and compare against strong baseline implementations. 
<!-- We also apply with the [roofline model](https://en.wikipedia.org/wiki/Roofline_model) when possible. -->

<!-- Items -->
## Benchmark Items

## Highlights
Compared with baselines, we hightlight inspiring performance achieved by Taichi on the basis of its easy-to-use programming style:
* Minimized coding efforts, doubled performance against CUDA in the [MPM benchmark](./mpm).
<p align="center">
<img src="mpm/fig/bench_2d.png" width="400">
</p>

* Comparable coding efforts, doubled performance against JAX in the [differentiable smoke simulation benchmark](./diff-taichi).
<p align="center">
<img src="diff-taichi/fig/bench_gpu.png" width="400">
</p>

* State-of-the-art performance, flexible coding style against CUDA in the [3x3 SVD benchmark](./svd3).
<p align="center">
<img src="svd3/fig/bench_svd.png" width="400">
</p>

* Performance approaches device capability roofline, in terms of both computation and memory bandwidth in the [Nested SAXPY Benchmark](./saxpy).
<p align="center">
<img src="saxpy/fig/roofline_log_scale.png" width="400">
</p>

Meanwhile, we have also noticed significant performance gap against manually optimized CUDA code in the [N-body benchmark](./n-body). 
<p align="center">
<img src="n-body/fig/bench_roofline.png" width="400">
</p>
It also occurs while comparing different backends in the [2D stencil benchmark](./stencil2d).
<p align="center">
<img src="stencil2d/fig/bench.png" width="300">
<img src="stencil2d/fig/bench_cuda_backend.png" width="300">
</p>
<p align="center">
<em>Taichi Performance gap with different backends. Left: Vulkan. Right: CUDA.</em>
</p>

We are now working on the compiler to shrink the gap.

## Contribution Guideline