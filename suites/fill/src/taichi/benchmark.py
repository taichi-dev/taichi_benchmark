from .fill import fill as run_fill

def benchmark():
    results = []
    N = 1024 * 1024 * 2
    for k in range(8):
        results.append(run_fill(N))
        N *= 2
    return {'taichi_fill':results}

if __name__ == '__main__':
    print(benchmark())
