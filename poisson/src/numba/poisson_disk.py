import numpy as np
from numpy import cos, sin, pi
import numba

def run_poisson(desired_samples = 100000):
    random = np.random.random
    grid_n = 400
    res = (grid_n, grid_n)
    dx = 1 / res[0]
    inv_dx = res[0]
    radius = dx * np.sqrt(2)
    
    grid = -np.ones(res, dtype=np.int32)
    samples = np.zeros((desired_samples, 2), dtype=float)
    active_samples = np.zeros((desired_samples * 2, 2), dtype=float)
    
    delta = (radius - 1e-6)**2
    
    
    @numba.jit(nopython=True)
    def coords_to_index(coords):
        x, y = (coords * inv_dx)
        return int(x), int(y)
    
    
    @numba.jit(nopython=True)
    def within_extent(p):
        x, y = p
        return 0 <= x < 1 and 0 <= y < 1
    
    
    @numba.jit(nopython=True)
    def generate_random_direction():
        theta = random() * 2 * pi
        return np.array([cos(theta), sin(theta)])
    
    
    @numba.jit(nopython=True)
    def generate_around_point(p):
        rad = (random() + 1) * radius
        return p + rad * generate_random_direction()
    
    
    @numba.jit(nopython=True)
    def check_collision(p, x, y, grid, samples):
        x0 = max(0, x - 2)
        x1 = min(grid_n, x + 3)
        y0 = max(0, y - 2)
        y1 = min(grid_n, y + 3)
        for i in range(x0, x1):
            for j in range(y0, y1):
                ind = grid[i, j]
                if ind != -1:
                    q = samples[ind]
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    if dx * dx + dy * dy < delta:
                        return True
        return False
    
    
    @numba.jit(nopython=True)
    def sample(grid, samples, active_samples):
        grid.fill(-1)
        num_sampled = 0
        active_samples[0] = 0.5
        tail = 1
        while tail > 0 and num_sampled < desired_samples:
            sample_index = int(random() * tail)
            source_x = active_samples[sample_index]
    
            generated = False
            for _ in range(100):
                new_x = generate_around_point(source_x)
                if within_extent(new_x):
                    ix, iy = coords_to_index(new_x)
                    collison = check_collision(new_x, ix, iy, grid, samples)
                    
                    if not collison and tail < desired_samples:
                        samples[num_sampled] = new_x
                        grid[ix, iy] = num_sampled
                        num_sampled += 1
    
                        active_samples[tail] = new_x
                        tail += 1
                        generated = True
                        break
            
            if not generated:
                tail -= 1
                active_samples[sample_index] = active_samples[tail]
        return num_sampled
    
    def run():
        import time
        # warm up
        sample(grid, samples, active_samples)
        repeats = 5
        
        t = time.perf_counter()
        for _ in range(repeats):
            sample(grid, samples, active_samples)
        avg_time_ms = (time.perf_counter() - t) / repeats * 1000
        return {'desired_samples': desired_samples, 'time_ms': avg_time_ms}
    return run()

if __name__ == '__main__':
    desired_samples = 1000
    print(run_poisson(desired_samples))


