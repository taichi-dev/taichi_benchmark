from numba import njit, prange, cuda
from utils import benchmark, nx, ny, dx, dy, dt, mu


@njit(parallel=True, fastmath=True)
def calc_velocity_numba(u, v, ut, vt):
    for i in prange(1, nx):
        for j in prange(1, ny+1):
            ut[i, j] = u[i, j] + dt * ((-0.25) *
                                       (((u[i+1, j] + u[i, j])**2 - (u[i, j] + u[i-1, j])**2) / dx
                                        + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j])
                                           - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy)
                                       + (mu) * ((u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dx ** 2
                                                 + (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dy ** 2))
    for i in prange(1, nx+1):
        for j in prange(1, ny):
            vt[i, j] = v[i, j] + dt * ((-0.25) *
                                       (((u[i, j+1] + u[i, j]) * (v[i+1, j]+v[i, j])
                                         - (u[i-1, j+1] + u[i-1, j]) * (v[i, j]+v[i-1, j])) / dx
                                        + ((v[i, j+1]+v[i, j]) ** 2-(v[i, j]+v[i, j-1]) ** 2) / dy)
                                       + (mu)*((v[i+1, j] - 2 * v[i, j] + v[i-1, j]) / dx ** 2
                                               + (v[i, j+1] - 2 * v[i, j] + v[i, j-1]) / dy ** 2))


@benchmark
def launch_numba(*args):
    calc_velocity_numba(*args)


@cuda.jit
def calc_velocity_numba_cuda(u, v, ut, vt):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid > ((nx + 1) * (ny + 1)):
        return
    i = tid // (ny + 1) + 1
    j = tid % (ny + 1) + 1
    if (i < nx and j < ny + 1):
        ut[i, j] = u[i, j] + dt * ((-0.25) *
                                   (((u[i+1, j] + u[i, j])**2 - (u[i, j] + u[i-1, j])**2) / dx
                                    + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j])
                                       - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy)
                                   + (mu) * ((u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dx ** 2
                                             + (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dy ** 2))
    if (i < nx + 1 and j < ny):
        vt[i, j] = v[i, j] + dt * ((-0.25) *
                                   (((u[i, j+1] + u[i, j]) * (v[i+1, j]+v[i, j])
                                     - (u[i-1, j+1] + u[i-1, j]) * (v[i, j]+v[i-1, j])) / dx
                                    + ((v[i, j+1]+v[i, j]) ** 2-(v[i, j]+v[i, j-1]) ** 2) / dy)
                                   + (mu)*((v[i+1, j] - 2 * v[i, j] + v[i-1, j]) / dx ** 2
                                           + (v[i, j+1] - 2 * v[i, j] + v[i, j-1]) / dy ** 2))


@benchmark
def launch_numba_cuda(*args):
    block_dim = 128
    total_threads = (nx + 1) * (ny + 1)
    grid_dim = (total_threads + block_dim - 1) // block_dim
    calc_velocity_numba_cuda[grid_dim, block_dim](*args)
    cuda.synchronize()
