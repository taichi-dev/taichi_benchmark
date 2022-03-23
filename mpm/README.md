# Material Point Method (MPM) Benchmark

## Introduction
Material Point Method (MPM) is widely used in physical simulations. The required 
computations for high-quality scenarios are intensive, and the achieved performance
is critical for real-time applications.
In this benchmark, we compare a simple implementation of the MPM method in Taichi
to its ported version in CUDA.

## Implementation
The measured CUDA implementation is based on an open-source implementation of an
[MPM3D](https://github.com/Aisk1436/mpm3d) (also for 2D computations) that was 
originally written in [Taichi](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm3d.py). For example, the <em>Particle to Grid</em> 
function is realized by an outermost for-loop within a Taichi kernel:

```python
for p in x:
    Xp = x[p] / dx
    base = int(Xp - 0.5)
    fx = Xp - base
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
    stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
    affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
    for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
        dpos = (offset - fx) * dx
        weight = 1.0
        for i in ti.static(range(dim)):
            weight *= w[offset[i]][i]
        grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
        grid_m[base + offset] += weight * p_mass
```
And the equivalent CUDA implementation becomes:

```c
__global__ void particle_to_grid_kernel(Vector *x, Vector *v, Matrix *C,
                                        const Real *J, Vector *grid_v,
                                        Real *grid_m, Real dx, Real p_vol,
                                        Real p_mass, int n_grid) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  Vector Xp = x[idx] / dx;
  Vectori base = (Xp.array() - 0.5).cast<int>();
  Vector fx = Xp - base.cast<Real>();
  std::array<Vector, 3> w{0.5 * (1.5 - fx.array()).pow(2),
                          0.75 - (fx.array() - 1.0).pow(2),
                          0.5 * (fx.array() - 0.5).pow(2)};
  auto stress = -dt * 4 * E * p_vol * (J[idx] - 1) / std::pow(dx, 2);
  Matrix affine = Matrix::Identity() * stress + p_mass * C[idx];
  for (auto offset_idx = 0; offset_idx < neighbour; offset_idx++) {
    Vectori offset = get_offset(offset_idx);
    Vector dpos = (offset.cast<Real>() - fx) * dx;
    Real weight = 1.0;
    for (auto i = 0; i < dim; i++) {
      weight *= w[offset[i]][i];
    }
    Vector grid_v_add = weight * (p_mass * v[idx] + affine * dpos);
    auto grid_m_add = weight * p_mass;
    Vectori grid_idx_vector = base + offset;
    auto grid_idx = 0;
    for (auto i = 0; i < dim; i++) {
      grid_idx = grid_idx * n_grid + grid_idx_vector[i];
    }
    for (auto i = 0; i < dim; i++) {
      atomicAdd(&(grid_v[grid_idx][i]), grid_v_add[i]);
    }
    atomicAdd(&(grid_m[grid_idx]), grid_m_add);
  }
}

```
Instead of explicitly mapping the problem size to each GPU thread, Taichi 
automatically parallelize the outermost for-loop. Consequently, the code
is more concise and easier to read.

## Evaluation
We conduct performance evaluation on the following device.

|Device| Nvidia RTX 3080 (10GB)|
|-----|-----------------------|
|FP32 performance| 29700 GFLOPS|
|Memory bandwidth| 760 GB/s|
|L2 cache capacity| 4MB|

<p align="center">
<img src="fig/bench_2d.png" width="560">
</p>

<p align="center">
<img src="fig/bench_3d.png" width="560">
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
