import taichi as ti
import time

ti.init(arch=ti.cuda, print_ir=False, log_level=ti.INFO, print_kernel_nvptx=False)

def run_nbody(nBodies):
    softening = 1e-5
    dt = 0.01
    nIters = 10
    block_size = 128

    velocities = ti.field(dtype=ti.float32)
    bodies = ti.field(dtype=ti.float32)
    ti.root.dense(ti.ijk, (nBodies // block_size, block_size, 4)).place(bodies)
    ti.root.dense(ti.ijk, (nBodies // block_size, block_size, 4)).place(velocities)


    @ti.kernel
    def randomizeBodies():
        for i, j, k in bodies:
            bodies[i, j, k] = 2.0 * ti.random(dtype=ti.float32) - 1.0
        for i, j, k in velocities:
            velocities[i, j, k] = 2.0 * ti.random(dtype=ti.float32) - 1.0

    @ti.kernel
    def bodyForce():
        ti.block_dim(block_size)
        for i in range(nBodies):
            Fx = 0.0
            Fy = 0.0
            Fz = 0.0
            X = bodies[i // block_size, i % block_size, 0]
            Y = bodies[i // block_size, i % block_size, 1]
            Z = bodies[i // block_size, i % block_size, 2]
            for j in range(nBodies // block_size):
                unroll_factor = 4
                for k in range(block_size / unroll_factor):
                    for t in ti.static(range(4)):
                        dx = bodies[j, k * unroll_factor + t, 0] - X
                        dy = bodies[j, k * unroll_factor + t, 1] - Y
                        dz = bodies[j, k * unroll_factor + t, 2] - Z
                        distSqr = dx * dx + dy * dy + dz * dz + softening
                        invDist = 1.0 / ti.sqrt(distSqr)
                        invDist3 = invDist * invDist * invDist
                        Fx += dx * invDist3
                        Fy += dy * invDist3
                        Fz += dz * invDist3
            velocities[i // block_size, i % block_size, 0] += dt * Fx;
            velocities[i // block_size, i % block_size, 1] += dt * Fy;
            velocities[i // block_size, i % block_size, 2] += dt * Fz;

        for i in range(nBodies):
            bodies[i // block_size, i % block_size, 0] += velocities[i // block_size, i % block_size, 0] * dt;
            bodies[i // block_size, i % block_size, 1] += velocities[i // block_size, i % block_size, 1] * dt;
            bodies[i // block_size, i % block_size, 2] += velocities[i // block_size, i % block_size, 2] * dt;

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
        #print("Finishing...time {}ms".format(avg_time))
        #print("nbodies={} speed {:.3f} billion bodies per second.".format(nBodies, 1e-6 * nBodies * nBodies / avg_time))
        return avg_time, 1e-6 * nBodies * nBodies / avg_time
    return run()

if __name__ == '__main__':
    nBodies = 1024
    repeats = 1
    for i in range(10):
        acc_time = 0.0
        for i in range(repeats):
            avg_time, _ = run_nbody(nBodies)
            acc_time = acc_time + avg_time
        print("{}, {:.3f}".format(nBodies, 1e-6 * nBodies * nBodies / acc_time * repeats))
        nBodies *= 2
        break
