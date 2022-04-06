import taichi as ti
import os
import math

def run_poisson(desired_samples = 100000):
    ti.init(arch=ti.cpu)
    
    run_test = False
    output_frames = False
    
    grid_n = 400
    res = (grid_n, grid_n)
    dx = 1 / res[0]
    inv_dx = res[0]
    radius = dx * math.sqrt(2)
    
    grid = ti.field(dtype=int, shape=res)
    samples = ti.Vector.field(2, dtype=float, shape=desired_samples)
    
    
    @ti.kernel
    def random_sample(desired_samples: int):
        if True:
            for i in range(desired_samples):
                samples[i] = ti.Vector([ti.random(), ti.random()])
    
    
    @ti.func
    def place_sample(sample_id, x):
        grid_index = int(ti.floor(x * inv_dx))
        grid[grid_index] = sample_id
    
    
    @ti.kernel
    def poisson_disk_sample(desired_samples: int) -> int:
        samples[0] = ti.Vector([0.5, 0.5])
        grid[int(grid_n * 0.5), int(grid_n * 0.5)] = 0
        head, tail = 0, 1
        # This kernel is serial since Taichi does not automatically parallelize top-level while loops
        while head < tail and head < desired_samples:
            source_x = samples[head]
            head += 1
    
            for repeats in range(100):
                theta = ti.random() * 2 * math.pi
                offset = ti.Vector([ti.cos(theta), ti.sin(theta)
                                    ]) * (1 + ti.random()) * radius
                new_x = source_x + offset
                new_index = ti.floor(new_x * inv_dx).cast(int)
    
                if 0 <= new_x[0] < 1 and 0 < new_x[1] < 1:
                    collision = False
                    for i in range(max(0, new_index[0] - 2),
                                   min(grid_n, new_index[0] + 3)):
                        for j in range(max(0, new_index[1] - 2),
                                       min(grid_n, new_index[1] + 3)):
                            if grid[i, j] != -1:
                                collision_x = samples[grid[i, j]]
                                if (collision_x - new_x).norm() < radius - 1e-6:
                                    collision = True
    
                    if not collision and tail < desired_samples:
                        samples[tail] = new_x
                        grid[new_index] = tail
                        tail += 1
        return tail
    
    
    @ti.kernel
    def test(num_samples: int):
        min_dist = 100.0
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                ti.atomic_min(min_dist, (samples[i] - samples[j]).norm())
        print('min distance=', min_dist, ' desired distance=', radius)
    
    
    def sample():
        grid.fill(-1)
        num_samples = poisson_disk_sample(desired_samples)
    
        if run_test:
            test(num_samples)
    
        return num_samples
    
    def run():
        import time
        sample()  # Warm up
        repeats = 5

        t = time.perf_counter()
        for i in range(repeats):
            sample()
        avg_time_ms = (time.perf_counter() - t) / repeats * 1000

        return {'desired_samples': desired_samples, 'time_ms': avg_time_ms}
    return run()

if __name__ == '__main__':
    desired_samples = 1000
    print(run_poisson(desired_samples))

