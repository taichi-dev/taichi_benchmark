import os
import json
from contextlib import contextmanager

from subprocess import Popen, PIPE

@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def run_binary(binary_file, argv):
    p = Popen(['./' + binary_file] + argv, stdout=PIPE)
    output, err = p.communicate()
    output = output.decode('utf-8')
    output = output.split('\n')
    results = []
    for line in output[:-1]:
        res_dict = None
        try:
            res_dict = json.loads(line)
        except:
            pass
        if res_dict:
            results.append(res_dict)
    return results

def compile_and_benchmark(source_name, output_binary_name, flags=[]):
    with pushd('src/cuda'):
        # Use pre-compiled binary
        results = []
        n_grids = [32, ]
        for n_grid in n_grids:
            argv = ["{}".format(n_grid), "{}".format(20)]
            results += run_binary(output_binary_name, argv)
        print("{} test finished.".format(output_binary_name))
        return results

def benchmark():
    return {"cuda_baseline": compile_and_benchmark("mpm3d.cu", 'build/MPM3D')}

if __name__ == '__main__':
    print(benchmark())
