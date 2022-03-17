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
        # Compile
        p = Popen(['nvcc', '-O3', source_name, '-DJSON_OUTPUT', '-o', output_binary_name] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot compile {}".format(output_binary_name))
        print("Successfully compiled {} into {}".format(source_name, output_binary_name))

        # Run Benchmark
        results = []
        N = 64
        for k in range(8):
            argv = ["{}".format(N)]
            results += run_binary(output_binary_name, argv)
            N *= 2
        print("{} test finished.".format(output_binary_name))
        return results

def benchmark():
    return {"cuda": compile_and_benchmark("stencil2d.cu", "stencil_cuda")}

if __name__ == '__main__':
    print(benchmark())
