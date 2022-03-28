import os
import json
import numpy as np
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

def compile_p2g(flags=[]):
    workdir = os.path.dirname(os.path.abspath(__file__))
    with pushd(workdir):
        # Compile
        p = Popen(['cmake', '-S', '.', '-B', 'build'] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot generate cmake{}".format(output_binary_name))

        p = Popen(['cmake', '--build', 'build', '--target', 'P2G'] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot compile {}".format(output_binary_name))
        print("Successfully compiled P2G")

def run_benchmark(output_binary_name, flags=[]):
    workdir = os.path.dirname(os.path.abspath(__file__))
    with pushd(workdir):
        # Run Benchmark
        results = []
        n_grids = np.arange(32, 256+32, 32).tolist()
        for n_grid in n_grids:
            print("running", output_binary_name, "n_grid", n_grid)
            argv = ["{}".format(n_grid)]
            results += run_binary(output_binary_name, argv)
        print("{} test finished.".format(output_binary_name))
        return results

def benchmark():
    compile_p2g()
    return {"cuda_baseline": run_benchmark('build/P2G')}

if __name__ == '__main__':
    print(benchmark())

