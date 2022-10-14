import suites
import logging

log = logging.getLogger(__name__)


def collect_benchmarks():
    benchmarks = []
    log.info('Collecting benchmarks...')
    for suite in suites.__all__:
        benchmarks += getattr(suites, suite).BENCHMARKS
    return benchmarks


def run_benchmarks():
    benchmarks = collect_benchmarks()
    log.info('Running benchmarks...')
    results = []
    for benchmark in benchmarks:
        results += benchmark.runner().run(benchmark())
    return results
