import taichi_benchmark as ti_bench

bench_items = [ti_bench.n_body, ti_bench.saxpy]

results = []
for item in bench_items:
    r = item.run()
    results.append(r)
print(results)
