# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark


# -- code --
class NBodyNaive(TaichiBenchmark):
    name = 'nbody'
    tags = {'variant': 'Naive'}
    archs = [ti.cuda, ti.vulkan]
    matrix = {
        'n': [128, 256, 512, 262144]
    }

    def init(self, n):
        self.n = n
        bodies = ti.field(shape=(n, 3), dtype=ti.float32)
        velocities = ti.field(shape=(n, 3), dtype=ti.float32)

        softening = 1e-9
        dt = 0.01

        @ti.kernel
        def randomizeBodies():
            for i, j in bodies:
                bodies[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0
            for i, j in velocities:
                velocities[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0

        @ti.kernel
        def bodyForce():
            ti.loop_config(block_dim=256)
            for i in range(n):
                Fx = 0.0
                Fy = 0.0
                Fz = 0.0
                for j in range(n):

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

            for i in range(n):
                bodies[i, 0] += velocities[i, 0] * dt
                bodies[i, 1] += velocities[i, 1] * dt
                bodies[i, 2] += velocities[i, 2] * dt

        randomizeBodies()
        self.bodyForce = bodyForce

    def run_iter(self):
        self.bodyForce()

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        return {
            'bips': self.n * self.n / avg_time / 1e9,
        }
