# Taichi Benchmarks

<!-- Purpose -->
The Taichi programming language is known for attaining high performance with easily understandable programs. The elegant parallel programming style has attracted many users to the Taichi community and we improve the compiler together. The benchmark codes here serve mainly three purposes:

* **Provide a target problem set**  
Since Taichi is a domain-specific language (DSL) focusing on the computer graphics and parallel computing domain, general benchmark cases cannot fully characterize Taichi to its benefit.
* **Provide a multi-dimensional comparison between Taichi and other popular frameworks**  
Performance is not the only objective, in fact, codes in this repository are not particularly tuned for the optimal performance. We also want to present the friendly, concise syntax Taichi exposed to its users. 
* **Open discussions for future performance improvements**  
Through comparing identical algorithms implemented in different frameworks, we can learn and benefit from the entire open-source community to keep improving our language and compiler.

In order to fulfill our purposes, we build this benchmark project with the following principles:

* **State-of-the-art baselines**  
Compare with well-performed baselines can help Taichi to get aware of further optimization opportunities.
* **Reproducible results**  
Tests can be reproduced with the `plot_benchmark.py` script under each subdirectory.
* **Easy-to-read coding style**  
Elegant coding style and high performance are equally important. Through comparisons between Taichi and manually optimized code, users can have a better understanding of Taichi's optimization techniques.

<!-- Items -->
<!-- ## Benchmark Items -->

<!-- #### Bare-metal performance benchmarks -->
<!-- #### Algorithm building blocks -->
<!-- #### Applications -->

<!-- results -->
## Highlights
We have conducted performance evaluation on an Nvidia Geforce RTX3080 graphics card. Compared with the baselines, we share some inspiring performance results achieved by Taichi on the basis of its easy-to-use programming style:

* Performance approaches device capability roofline, in terms of both computation and memory bandwidth. \[Source: [Nested SAXPY](./suites/saxpy), [Array fill](./suites/fill).\]

<p align="center">
<img src="suites/saxpy/fig/roofline_log_scale.png" width="400">  
<img src="suites/fill/fig/bench_fill.png" width="400">
</p>

* Minimized coding efforts, comparable performance against CUDA. \[Source: [MPM](./suites/mpm), [3x3 SVD](./suites/svd3), [Path Tracer](./suites/smallpt), [Nested SAXPY](./suites/saxpy).\]

<p align="center">
<img src="suites/mpm/fig/bench_3d.png" width="400" height="300">
<img src="suites/svd3/fig/bench_svd.png" width="400" height="300">
</p>

<p align="center">
<img src="suites/smallpt/fig/bench.png" width="400">
<img src="suites/saxpy/fig/compute_bench.png" width="400">
</p>
<p align="center">
  <img src="suites/n_body/fig/bench.png" width="400">
  <img src="suites/stencil2d/fig/bench.png" width="400">
</p>

* Easy-to-read code, extraordinary performance against JAX (GPU) and Numba/Numpy (CPU). \[Source: [Differentiable Smoke Simulation](./suites/diff-smoke), [Poisson Disk Sampling](./suites/poisson).\]

<p align="center">
<img src="suites/diff-smoke/fig/bench_gpu.png" width="400">
<img src="suites/poisson/fig/bench.png" width="400">
</p>

<!-- Future works -->
<!-- Contribution Guidelines -->
## Future Works
We are driving the benchmark work in two directions:

* **More use cases with strong baseline implementations**  
We are working on extending our benchmarks to cover more generalized parallel tasks. Benchmark items can be added when there are proper baseline implementations to compare with.

* **More target backends**  
The current tests are conducted primarily on Nvidia GPUs. We are extending our benchmark on more devices as Taichi is designed to be hardware neural. Also, performance reports are welcome if you have a supported device by Taichi!
