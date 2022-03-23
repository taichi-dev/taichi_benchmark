# N-Body Simulation Benchmark

## Introduction

The [N-Body simulation](ihttps://en.wikipedia.org/wiki/N-body_simulation) is an interesting problem in physical cosmology. 
In this benchmark, we take the most easily understandable direct method with O(N * N) complexity, which computes every pair of the body interactions.

## Implementation

The CUDA codes credit to Mark Harris's intuitive stepwise n-body optimization tutorial [mini-nbody](https://github.com/harrism/mini-nbody). 
We made minor changes to make it runnable in our code repository. 

Taichi's implementation is translated from the [C prototype](https://github.com/harrism/mini-nbody/blob/master/nbody.c), you can find direct mappings of the two code snippets. The C version of force calculation function looks like this:

```c
void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f; 
    float Fy = 0.0f; 
    float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;
      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }
    p[i].vx += dt * Fx; 
    p[i].vy += dt * Fy; 
    p[i].vz += dt * Fz;
  }
}
```

While the equivalent Taichi version is

```python
def bodyForce():
    for i in range(nBodies):
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        for j in range(nBodies):
            dx = bodies[j, 0] - bodies[i, 0]
            dy = bodies[j, 1] - bodies[i, 1]
            dz = bodies[j, 2] - bodies[i, 2]
            distSqr = dx * dx + dy * dy + dz * dz + softening
            invDist = 1.0 / ti.sqrt(distSqr)
            invDist3 = invDist * invDist * invDist
            Fx += dx * invDist3
            Fy += dy * invDist3
            Fz += dz * invDist3
        velocities[i, 0] += dt * Fx
        velocities[i, 1] += dt * Fy
        velocities[i, 2] += dt * Fz
```
Quite similar, right?

The CUDA equivalent version is a bit confusing if you have no prior knowlege about SIMT parallel programming. The outermost loop is implicitly replaced by parallelizaiton. 
```cuda
__global__
void bodyForce(Body *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f;
    float Fy = 0.0f; 
    float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;
      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }
    p[i].vx += dt * Fx; 
    p[i].vy += dt * Fy; 
    p[i].vz += dt * Fz;
  }
}
```

#### Block Access

By repeatedly loading a small block of data, it is possible to leverage the GPU's L1 data cache to improve performance.
In CUDA, we explicitly decompose the loop and access by blocks. 
```cuda
__global__
void bodyForce(Body *p, Body *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    ...
    for (int tile = 0; tile < gridDim.x; tile++) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
          ...
      }
      __syncthreads();
    }
    ...
  }
}
```

In Taichi, we only change the definition of fileds in order to enable block-wise memory access.

```python
ti.root.dense(ti.ij, (nBodies // block_size, 4)).dense(ti.i, block_size).place(bodies)
```

Taichi internally handles the loop decompositions and index permutations according to the block statements in field definitions.
There are no changes in the `bodyForce` function. If you are unformaliar with the usage, please refer to our document [Fields (Advanced)](https://docs.taichi.graphics/lang/articles/advanced/layout).

#### Partial Unrolling

We notice that the `nvcc` implicitly improves performance by unroll the inner loop by a factor of 4.
Taichi haven't integrate this optmization yet so we provide a version that unrolls explicitly.
The current code is a little hard to understand. We will improve the compiler to elegantly implement this behavior. Stay tuned! 

#### Other Optimizations

There are other optimization techniques used with CUDA, including the `float4` vector type, better shared memory support and others. Please refer to the source code for details.

## Evaluation

We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 5 MB|


Performance is measured with "Billion body interactions per second" to indicate compute throughtput. The performance of different Taichi and CUDA implementations is illustrated in the following figure.

<p align="center">
<img src="fig/bench_roofline.png" width="560">
</p>

In this figure, `Taichi/Baseline` and `CUDA/Baseline` refer to the very original implementation without any optimizations. `Taichi/Block` and `CUDA/Block` optimizes block-wise memory access. `Taichi/Unroll` explicitly unrolls the compute loop. `CUDA/Optimized` enables all optimization methods.

We observe that with the same level of optimizations, Taichi and CUDA can achieve similar performance:
* `Taichi/Baseline` slightly lags behind `CUDA/Baseline` due to `nvcc`'s partial unrolling optimiztion.
* `Taichi/Unroll` achieves comparable performance with `CUDA/Block` where they share same set of optimization techniques.
* `CUDA/Optimized` reveals overwhelming good performance for large body count. 

Given the first two positive results, we are confident that Taichi can match the best CUDA implementation by upgrading the compiler. 

<!-- issue tracker to call for contrib? -->

## Reproduction Steps

* Pre-requisites
```shell
python3 -m pip install --upgrade taichi
python3 -m pip install matplotlib
```
If you want to compare with CUDA, make sure you have `nvcc` properly installed.

* Run the benchmark and draw the plots
```shell
cd n-body
python3 plot_benchmark.py
```
