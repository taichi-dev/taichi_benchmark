# p2g from the mpm88 implementation 
import taichi as ti
from time import perf_counter

def run_p2g(n_grid = 128):
    ti.init(arch=ti.gpu)
    
    n_particles = n_grid**2 // 2
    dx = 1 / n_grid
    dt = 2e-4
    
    p_rho = 1
    p_vol = (dx * 0.5)**2
    p_mass = p_vol * p_rho
    E = 400
    
    x = ti.Vector.field(2, float, n_particles)
    v = ti.Vector.field(2, float, n_particles)
    C = ti.Matrix.field(2, 2, float, n_particles)
    J = ti.field(float, n_particles)
    grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
    grid_m = ti.field(float, (n_grid, n_grid))
    
    @ti.kernel
    def p2g():
        for p in x:
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(1, 1)):
                offset = ti.Vector([i, j])
                dpos = (offset - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
    
    @ti.kernel
    def init():
        for i in range(n_particles):
            x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
            v[i] = [0, -1]
            J[i] = 1
        for i, j in grid_m:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0

    def run():
        # skip first run
        init()
        for s in range(32):
            p2g()
            ti.sync()

        # measure 
        t_start = perf_counter()
        nIters = 1024
        for _ in range(nIters):
            for s in range(32):
                p2g()
                ti.sync()
        t_stop = perf_counter()
        return {'n_particles': n_particles, 'time_ms': (t_stop - t_start)*1000/nIters}
    return run()


if __name__ == '__main__':
    n_grid=128
    for _ in range(5):
        result = run_p2g(n_grid)
        n_particles = result['n_particles']
        time_ms = result['time_ms']
        print("{} particles run {:.3f} ms".format(n_particles, time_ms))

