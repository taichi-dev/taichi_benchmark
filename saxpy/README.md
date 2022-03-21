# SAXPY Benchmark
## Algorithm
The SAXPY (Single-precision AX Plus Y) implements the formula of `Y = aX + Y`, where `X` and `Y` are vectors and a is a constant coefficient. 
It is a standard algorithm implemented in almost all BLAS libraries. 
The SAXPY algorithm conducts one multiplication and one addition arithmetic on each element. 
In terms of memory accesses, it triggers 2 reads and 1 write operations. 
Therefore, the overall compute load is `N * 2` and memory access footprint is `N * 3 * sizeof(float) = 12 * N`. 
The arithmetic intensity is merely `1/6`, which is apparently bound by memory bandwidth.
In order to benchmark a more compute-intensive algorithm, we nest several SAXPY computations, that is to say compute 
`Y = az * (ay * ( .... aa*X + Y)  ... + Y) +Y)`.
Given the nesting factor `m`, the computing load becomes `m * N * 2` while the memory footprint remains the same. 
The arithmetic intensity becomes `m / 6`, which linearly increases with the nesting factor.

## Implementation

The Taichi's SAXPY kernel implementation is brief as follows:
```python
coeff=[1.0, 2.0, 3.0, 4.0]
@ti.kernel
def saxpy_kernel(x: ti.template(), y: ti.template()):
    for i in ti.grouped(y):
        # Statically unroll
        z_c = x[i]
        for c in ti.static(coeff):
            z_c = c * z_c + y[i]
        y[i] = z_c
```
The values in `coeff` correponds to the constant coefficient `a` in the formulas. The length of `coeff` array corresponds to the nesting factor `m`. 
When `coeff` holds only one element, the implementation degrades to standard SAXPY.
We take two different CUDA-based approaches as baselines: Thrust and cuBLAS.  
The details can be found in [Nvidia's official blog](https://developer.nvidia.com/blog/six-ways-saxpy/). 
For nested SAXPY, however, cuBLAS cannot efficiently leverage the increased arithmetic intensity as it is static pre-compiled. 
Thrust can nest by manually composing the formulas, a little troublesome though.

## Evaluation
Device Specification

|Model| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 4MB|

### Standard SAXPY
We benchmark standard SAXPY on matrix shapes from `256x256` to `4096x4096`. 
The cuBLAS reveals great results for small matrices. When matrix shapes grow larger than L2 cache capacity, all implementations perform similarly.

![compute_bench](saxpy/fig/compute_bench.png)


### Nested SAXPY
By nesting multiple SAXPY routines, we drastically increase arithmetic load with respect to the same memory footprint. Taichi significantly outperforms Thrust while keeping concise programming styles. By contrasting Taichi/Thrust and cuBLAS, we can conclude that flexibility sometimes delivers significant speedup, especially when dealing with complex problems.

![nesting_bench_512](saxpy/fig/nesting_bench_512.png)
![nesting_bench_4096](saxpy/fig/nesting_bench_4096.png)



### Roofline
In the previous section, we have greatly increased computing performance by nesting multiple SAXPY together. A question arises here: if we keep on increasing the nesting factor, how good can Taichi perform on the GPU?
Below is the roofline plot of benchmark results on `4096x4096` arrays. 
The points represent varying nesting factors. 
We have two conclusions: 
* As nesting factor increases, the problem transists from bandwidth-bound type to compute-bound type.
* In either case, Taichi can fully unleash the GPU's computing power.

![roofline_log_scale](saxpy/fig/roofline_log_scale.png)

## Reproduction steps
-
## Conclusion
-
