from src.taichi.stencil2d import run_stencil

def benchmark():
    N = 64
    results = []
    for k in range(8):
        results.append(run_stencil(N))
        N *= 2
    return {"taichi": results}

if __name__ == "__main__":
    print(benchmark())
