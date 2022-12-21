# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark


# -- code --
class MPM(TaichiBenchmark):
    name = 'mpm'
    archs = [ti.cuda, ti.vulkan, ti.metal]
    matrix = {
        'n_grid': [32, 64, 128, 256],
        'dim' : [2, 3]
    }

    @property
    def repeats(self):
        if self.n_grid > 64:
            return 10
        return 40


    def init(self, n_grid, dim):
        assert dim in (2, 3)
        steps, dt = 25, 1e-4
        if dim == 3:
            dt = 8e-5
        n_particles = n_grid ** dim // 2 ** (dim - 1)
        dx = 1 / n_grid

        p_rho = 1
        p_vol = (dx * 0.5)**2
        p_mass = p_vol * p_rho
        gravity = 9.8
        bound = 3
        E = 400

        x = ti.Vector.field(dim, float, n_particles)
        v = ti.Vector.field(dim, float, n_particles)
        C = ti.Matrix.field(dim, dim, float, n_particles)
        J = ti.field(float, n_particles)

        grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
        grid_m = ti.field(float, (n_grid, ) * dim)

        neighbour = (3, ) * dim

        @ti.kernel
        def init():
            for i in range(n_particles):
                x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15
                J[i] = 1

        @ti.kernel
        def mpm_substep():
            for I in ti.grouped(grid_m):
                grid_v[I] = ti.zero(grid_v[I])
                grid_m[I] = 0
            ti.loop_config(block_dim=64)
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
                    grid_v[base +
                           offset] += weight * (p_mass * v[p] + affine @ dpos)
                    grid_m[base + offset] += weight * p_mass 
            for I in ti.grouped(grid_m):
                if grid_m[I] > 0:
                    grid_v[I] /= grid_m[I]
                grid_v[I][1] -= dt * gravity
                cond = (I < bound) & (grid_v[I] < 0) | \
                       (I > n_grid - bound) & (grid_v[I] > 0)
                grid_v[I] = ti.select(cond, 0, grid_v[I])

            ti.loop_config(block_dim=64)
            for p in x:
                Xp = x[p] / dx
                base = int(Xp - 0.5)
                fx = Xp - base
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
                new_v = ti.zero(v[p])
                new_C = ti.zero(C[p])
                for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                    dpos = (offset - fx) * dx
                    weight = 1.0
                    for i in ti.static(range(dim)):
                        weight *= w[offset[i]][i]
                    g_v = grid_v[base + offset]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
                v[p] = new_v
                x[p] += dt * v[p]
                J[p] *= 1 + dt * new_C.trace()
                C[p] = new_C

        init()
        self.mpm_substep = mpm_substep        
        self.steps = steps
        self.n_particles = n_particles
        self.dim = dim
        self.n_grid = n_grid

    def run_iter(self):
        for _ in range(self.steps):
            self.mpm_substep()

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        return {
            'fps': 1.0 / avg_time,
            'particles/s': self.n_particles * self.steps / avg_time
        }
