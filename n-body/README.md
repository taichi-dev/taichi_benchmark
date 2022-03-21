# N-Body Simulation Benchmark

## Introduction

The [N-Body simulation](ihttps://en.wikipedia.org/wiki/N-body_simulation) is an interesting problem in physical cosmology. 
In this benchmark, we take the most easily understandable direct method with O(N * N) complexity, as we only care about compute performance.

## Implementation

The CUDA code should credit to Mark Harris's intuitive stepwise n-body optimization tutorial [mini-nbody](https://github.com/harrism/mini-nbody). 
We made minor changes to make it runnable in our code repository.

Taichi's implementation is directly translated from the [C prototype](https://github.com/harrism/mini-nbody/blob/master/nbody.c), where you can find direct mappings of the two code snippets. For example, the C version of force calculation function looks like this:

```c
void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
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
Quite similar, right? Then let's take a look at the performance!

## Evaluation

## Reproduction Steps
