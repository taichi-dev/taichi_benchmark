# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark


# -- code --

class NBodyCacheBlock(TaichiBenchmark):
    name = 'nbody'
    tags = {'variant': 'CacheBlock'}
    archs = [ti.cuda, ti.vulkan]
    matrix = {
        'n': [128, 256, 512, 262144]
    }

    def init(self, n):
        self.n = n

        softening = 1e-9
        dt = 0.01
        block_size = 128

        velocities = ti.field(dtype=ti.float32)
        bodies = ti.field(dtype=ti.float32)
        ti.root.dense(ti.ij, (n, 4)).place(velocities)
        ti.root.dense(ti.ij, (n, 4)).place(bodies)

        @ti.kernel
        def randomizeBodies():
            for i, j in bodies:
                bodies[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0
            for i, j in velocities:
                velocities[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0

        @ti.kernel
        def bodyForce():
            ti.loop_config(block_dim=block_size)
            for i in range(n):
                Fx = 0.0
                Fy = 0.0
                Fz = 0.0
                body_i_x = bodies[i, 0]
                body_i_y = bodies[i, 1]
                body_i_z = bodies[i, 2]
                pad = ti.simt.block.SharedArray((block_size, 3), ti.f32)
                g_tid = ti.simt.block.global_thread_idx()
                tid = g_tid % block_size
                for k in range(n // block_size):
                    # Load into shared memory
                    pad[tid, 0] = bodies[k * block_size + tid, 0]
                    pad[tid, 1] = bodies[k * block_size + tid, 1]
                    pad[tid, 2] = bodies[k * block_size + tid, 2]
                    ti.simt.block.sync()
                    for j in range(block_size):
                        dx = pad[j, 0] - body_i_x
                        dy = pad[j, 1] - body_i_y
                        dz = pad[j, 2] - body_i_z
                        distSqr = dx * dx + dy * dy + dz * dz + softening
                        invDist = 1.0 / ti.sqrt(distSqr)
                        invDist3 = invDist * invDist * invDist
                        Fx += dx * invDist3
                        Fy += dy * invDist3
                        Fz += dz * invDist3
                    ti.simt.block.sync()
                velocities[i, 0] += dt * Fx
                velocities[i, 1] += dt * Fy
                velocities[i, 2] += dt * Fz

            for i in range(n):
                bodies[i, 0] = bodies[i, 0] + velocities[i, 0] * dt
                bodies[i, 1] = bodies[i, 1] + velocities[i, 1] * dt
                bodies[i, 2] = bodies[i, 2] + velocities[i, 2] * dt

        randomizeBodies()
        self.bodyForce = bodyForce

    def run_iter(self):
        self.bodyForce()

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        return {
            'bips': self.n * self.n / avg_time / 1e9,
        }

if __name__ == '__main__':
    ti.init(arch=ti.cuda, log_level=ti.TRACE, offline_cache=False, print_kernel_nvptx=True)
    runner=NBodyCacheBlock()
    runner.init(32768)
    runner.run_iter()

    import time
    st = time.perf_counter()
    for i in range(10):
        runner.run_iter()
    et = time.perf_counter()
    t = (et - st) / 10
    print(t)

    print(runner.get_metrics(t))
