# N-Body Simulation Benchmark

## Introduction

The [N-Body simulation](ihttps://en.wikipedia.org/wiki/N-body_simulation) is an interesting problem in physical cosmology. 
In this benchmark, we take the most easily understandable direct method with O(N * N) complexity, as we only care about compute performance.

## Implementation

The CUDA code should credit to Mark Harris's intuitive stepwise n-body optimization tutorial [mini-nbody](https://github.com/harrism/mini-nbody). 
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
        velocities[i, 0] += dt * Fx;
        velocities[i, 1] += dt * Fy;
        velocities[i, 2] += dt * Fz;
```
The CUDA equivalent verion is a bit confusing if you have no prior knowlege about SIMT parallel programming. The outermost parallel loop is implicitly replaced by parallelizaiton. Even though, it is still an intuitive implementation.
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

### Blocked Access

By repeatively loading a small block of data, it is possible to leverage the L1 data cache to improve performance.
In CUDA programming, we explicitly decompose the loop and access by blocks. 
```cuda
__global__
void bodyForce(Body *p, Body *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; 
    float Fy = 0.0f; 
    float Fz = 0.0f;
    for (int tile = 0; tile < gridDim.x; tile++) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
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
      __syncthreads();
    }
    v[i].x += dt*Fx;
    v[i].y += dt*Fy;
    v[i].z += dt*Fz;
  }
}
```

In Taichi, only the definition of fileds is changed to enable block-wise memory access.

```python
ti.root.dense(ti.ij, (nBodies // block_size, 4)).dense(ti.i, block_size).place(bodies)
```

### Explicit Unrolling

We further improve Tachi's performance by explicitly unroll a dimension by a factor of 2 or four. 
The code is a little complex. 
We won't dive into details here as we will improve the compiler to implicitly implement this behavior. Stay tuned! 

### Others

There are other optimization techniques used with CUDA, including the `float4` vector type, better shared memory support and others. Please refer to the source code for details.

## Evaluation

We illustrate performance of different Taichi and CUDA implementations.

<p align="center">
<img src="fig/bench_roofline.png" width="560">
</p>

## Reproduction Steps
