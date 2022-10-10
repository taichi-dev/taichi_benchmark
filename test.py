import taichi_benchmark as ti_bench
import os

# bench_items = [ti_bench.n_body, ti_bench.saxpy]
bench_items = [ti_bench.n_body]

results = []
for item in bench_items:
    r = item.run()
    if type(r) in [list, tuple]:
        results += r
    else:
        results.append(r)
print(results)

auth = os.getenv("TI_BENCH_AUTH")
ti_bench.common.upload_benchmark_results(results, auth)