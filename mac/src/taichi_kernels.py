import taichi as ti
from utils import benchmark, nx, ny, dx, dy, dt, mu


def init_taichi():
    ti.init(arch=ti.cuda, device_memory_fraction=0.5)


@ti.kernel
def calc_velocity_taichi(u: ti.types.ndarray(), v: ti.types.ndarray(), ut: ti.types.ndarray(), vt: ti.types.ndarray()):
    for i, j in ti.ndrange((1, nx), (1, ny+1)):
        ut[i, j] = u[i, j] + dt * ((-0.25) *
                                   (((u[i+1, j] + u[i, j])**2 - (u[i, j] + u[i-1, j])**2) / dx
                                    + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j])
                                       - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy)
                                   + (mu) * ((u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dx ** 2
                                             + (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dy ** 2))
    for i, j in ti.ndrange((1, nx+1), (1, ny)):
        vt[i, j] = v[i, j] + dt * ((-0.25) *
                                   (((u[i, j+1] + u[i, j]) * (v[i+1, j]+v[i, j])
                                     - (u[i-1, j+1] + u[i-1, j]) * (v[i, j]+v[i-1, j])) / dx
                                    + ((v[i, j+1]+v[i, j]) ** 2-(v[i, j]+v[i, j-1]) ** 2) / dy)
                                   + (mu)*((v[i+1, j] - 2 * v[i, j] + v[i-1, j]) / dx ** 2
                                           + (v[i, j+1] - 2 * v[i, j] + v[i, j-1]) / dy ** 2))


@benchmark
def launch_taichi(*args):
    calc_velocity_taichi(*args)
    ti.sync()
