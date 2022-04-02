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
    workdir = os.path.dirname(os.path.abspath(__file__))
    with pushd(workdir):
        # Compile
        p = Popen(['nvcc', '-O3', source_name, '-DJSON_OUTPUT', '-o', output_binary_name] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot compile {}".format(output_binary_name))
        print("Successfully compiled {} into {}".format(source_name, output_binary_name))

        # Run Benchmark
        results = []
        N = 8192
        nIter = 10
        for k in range(10):
            argv = ["{} {}".format(N, nIter)]
            results += run_binary(output_binary_name, argv)
            N *= 2
        print("{} test finished.".format(output_binary_name))
        return results

def benchmark():
    workdir = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.join(workdir, "3x3_SVD_CUDA/svd3x3/svd3x3/")
    if not os.path.exists(incdir):
        raise ValueError("Include dir not found {}".format(incdir))
    print(incdir)
    return {"cuda_baseline": compile_and_benchmark("benchmark.cu", "svd_cuda_baseline", flags=["-I"+incdir]),
            "cuda_shared": compile_and_benchmark("benchmark.cu", "svd_cuda_shared", flags=["-I"+incdir, "-DUSE_SHARED"])}

if __name__ == '__main__':
    print(benchmark())
