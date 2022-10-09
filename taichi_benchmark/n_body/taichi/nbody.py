import taichi as ti
from taichi_benchmark.common import benchmark

@benchmark(test_name='N-Body baseline', archs=[ti.cuda, ti.vulkan], repeats=20)
def nbody(**kwargs):
    nBodies = kwargs['nBodies']
    softening = 1e-9
    dt = 0.01

    bodies = ti.field(shape=(nBodies,  3), dtype=ti.float32)
    velocities = ti.field(shape=(nBodies, 3), dtype=ti.float32)

    @ti.kernel
    def randomizeBodies():
        for i, j in bodies:
            bodies[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0
        for i, j in velocities:
            velocities[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0

    @ti.kernel
    def bodyForce():
        ti.loop_config(block_dim=256)
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

        for i in range(nBodies):
            bodies[i, 0] += velocities[i, 0] * dt
            bodies[i, 1] += velocities[i, 1] * dt
            bodies[i, 2] += velocities[i, 2] * dt
    
    def benchmark_init():
        randomizeBodies()
        bodyForce()
    
    def benchmark_iter():
        bodyForce()
    
    def benchmark_metrics(avg_time):
        return {'rate': 1e-9 * nBodies * nBodies / avg_time}

    return benchmark_init, benchmark_iter, benchmark_metrics
