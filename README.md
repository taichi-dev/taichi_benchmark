# Taichi Benchmarks

<!-- Purpose -->
The Taichi programming language is known for attaining high performance with easily understandable programs. The elegant parallel programming style has attracted many users to the Taichi community and we improve the compiler together. The benchmark codes here serve mainly three purposes:

* **Provide a target problem set**  
Since Taichi is a domain-specific language (DSL) focusing on the computer graphics and parallel computing domain, general benchmark cases cannot fully characterize Taichi to its benefit.
* **Provide a multidimensional comparison between Taichi and other popular frameworks**  
Performance is not the only objective, in fact, codes in this repository are not particularly tuned for the optimal performance. We also want to present the friendly, concise syntax Taichi exposed to its users. 
* **Open discussions for future performance improvements**  
Through comparing identical algorithms implemented in different frameworks, we can learn and benefit from the entire open-source community to keep improving our language and compiler.

In order to fulfill our purposes, we build this benchmark project with the following principles:
* **Strong baselines**  
* **Reproducibile Results**  
* **Readable Coding Style**

<!-- Items -->
## Benchmark Items

#### Bare-metal performance benchmarks
#### Algorithm building blocks
#### Applications

<!-- results -->
## Highlights
Compared with baselines, we hightlight inspiring performance achieved by Taichi on the basis of its easy-to-use programming style:
* Minimized coding efforts, doubled performance against CUDA and JAX in the [MPM](./mpm) and [differentiable smoke simulation](./diff-smoke) benchmarks.
<p align="center">
<img src="mpm/fig/bench_2d.png" width="400">
<img src="diff-smoke/fig/bench_gpu.png" width="400">
</p>


* State-of-the-art performance, flexible coding style against CUDA in the [3x3 SVD](./svd3) and [path tracer](./smallpt) benchmarks.

<p align="center">
<img src="svd3/fig/bench_svd.png" width="400">
<img src="smallpt/fig/bench.png" width="400">
</p>

* Performance approaches device capability roofline, in terms of both computation and memory bandwidth in the [nested SAXPY](./saxpy) and [array fill](./fill) benchmarks.
<p align="center">
<img src="saxpy/fig/roofline_log_scale.png" width="400">
<img src="fill/fig/bench_fill.png" width="400">
</p>

# Gaps

We have also noticed significant performance gap against manually optimized CUDA code in the [N-body](./n-body) and [2D stencil](./stencil2d) benchmarks. 
We are working on the Taichi compiler to shrink the performance gaps. Welcome contribution!
<p align="center">
<img src="n-body/fig/bench_roofline.png" width="400">
<img src="stencil2d/fig/bench_cuda_backend.png" width="400">
</p>

<!-- Future works -->
<!-- Contribution Guidelines -->
## Future Works
We are pushing the benchmark work in three directions:
* **Extend more benchmark items**  
Placeholder
* **Build stronger baselines**  
Placeholder
* **Benchmark on more devices**  
Placeholder
