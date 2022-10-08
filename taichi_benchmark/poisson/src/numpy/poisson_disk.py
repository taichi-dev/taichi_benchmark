import numpy as np
from numpy import cos, sin, pi

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
    
    
    def generate_random_direction(num):
        theta = random(num) * 2 * pi
        return cos(theta), sin(theta)
    
    
    def generate_around_point(p, num):
        rad = (random(num) + 1) * radius
        lx, ly = generate_random_direction(num)
        qx = p[0] + rad * lx
        qy = p[1] + rad * ly
        c1 = (qx >= 0) & (qx <= 1)
        c2 = (qy >= 0) & (qy <= 1)
        cond = c1 & c2
        qx = qx[cond]
        qy = qy[cond]
        ix = (qx * inv_dx)
        iy = (qy * inv_dx)
        return np.stack((qx, qy, ix, iy), axis=1)
    
    
    def check_collision(px, py, x, y):
        x0 = max(0, x - 2)
        x1 = min(grid_n, x + 3)
        y0 = max(0, y - 2)
        y1 = min(grid_n, y + 3)
        for i in range(x0, x1):
            for j in range(y0, y1):
                ind = grid[i, j]
                if ind != -1:
                    q = samples[ind]
                    dx = px - q[0]
                    dy = py - q[1]
                    if dx * dx + dy * dy < delta:
                        return True
        return False
    
    
    def sample():
        grid.fill(-1)
        num_sampled = 0
        active_samples[0] = 0.5
        tail = 1
        while tail > 0 and num_sampled < desired_samples:
            sample_index = int(random() * tail)
            source_x = active_samples[sample_index]
    
            generated = False
            pts = generate_around_point(source_x, 100)
            for qx, qy, ix, iy in pts:
                ix, iy = int(ix), int(iy)
                collison = check_collision(qx, qy, ix, iy)
    
                if not collison and tail < desired_samples:
                    samples[num_sampled] = (qx, qy)
                    grid[ix, iy] = num_sampled
                    num_sampled += 1
    
                    active_samples[tail] = (qx, qy)
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
        sample()
        repeats = 5

        t = time.perf_counter()
        for _ in range(repeats):
            sample()
        avg_time_ms = (time.perf_counter() - t) / repeats * 1000
        return {'desired_samples': desired_samples, 'time_ms': avg_time_ms}
    return run()

if __name__ == '__main__':
    desired_samples = 1000
    print(run_poisson(desired_samples))

