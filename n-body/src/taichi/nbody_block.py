import taichi as ti
import time

def run_nbody(nBodies, arch=ti.cuda, nIters=50):
    ti.init(arch=arch)
    softening = 1e-9
    dt = 0.01
    block_size = 128

    velocities = ti.field(dtype=ti.float32)
    bodies = ti.field(dtype=ti.float32)
    ti.root.dense(ti.ij, (nBodies, 4)).place(velocities)
    ti.root.dense(ti.ij, (nBodies // block_size, 4)).dense(ti.i, block_size).place(bodies)

    @ti.kernel
    def randomizeBodies():
        for i, j in bodies:
            bodies[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0
        for i, j in velocities:
            velocities[i, j] = 2.0 * ti.random(dtype=ti.float32) - 1.0

    @ti.kernel
    def bodyForce():
        ti.block_dim(block_size)
        ti.block_local(bodies)
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
            velocities[i, 0] = velocities[i, 0] + dt * Fx;
            velocities[i, 1] = velocities[i, 1] + dt * Fy;
            velocities[i, 2] = velocities[i, 2] + dt * Fz;

        for i in range(nBodies):
            bodies[i, 0] = bodies[i, 0] + velocities[i, 0] * dt;
            bodies[i, 1] = bodies[i, 1] + velocities[i, 1] * dt;
            bodies[i, 2] = bodies[i, 2] + velocities[i, 2] * dt;

    def run():
        randomizeBodies()
        st = None
        for i in range(nIters):
            bodyForce()
            ti.sync()
            if st == None:
                st = time.time()
        et = time.time()

        avg_time =  (et - st) * 1000.0 / (nIters- 1)
        return {'nbodies': nBodies, 'time': avg_time, 'rate': 1e-6 * nBodies * nBodies / avg_time}
    return run()

if __name__ == '__main__':
    nBodies = 1024
    for i in range(10):
        result = run_nbody(nBodies)
        avg_time = result['time']
        print("nBodies={}, spped {:.3f} billion bodies per second.".format(nBodies, 1e-6 * nBodies * nBodies / avg_time))
        nBodies *= 2
