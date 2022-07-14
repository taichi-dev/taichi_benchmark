import os
import sys
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

def run_benchmark(binary_name, num, repeats=5):
    # Run Benchmark
    workdir = os.path.dirname(os.path.abspath(__file__))
    atime = 0
    niter = 0
    while(niter < repeats):
        p = Popen([workdir+'/'+binary_name, str(num)], stdout=PIPE)
        output, err = p.communicate()
        output = output.decode('utf-8')
        atime += float(output)
        niter += 1
    atime /= repeats
    return str(atime)

def benchmark(scale=list):
    compile_and_benchmark("scan_cuda.cu", "cuda")
    compile_and_benchmark("scan_cub.cu", "cub")
    compile_and_benchmark("scan_thrust.cu", "thrust")
    results_cuda = []
    results_cub = []
    results_thrust = []
    for num in scale:
        results_cuda.append(run_benchmark("cuda", num))
        results_cub.append(run_benchmark("cub", num))
        results_thrust.append(run_benchmark("thrust", num))
    return {"cuda": results_cuda,
            "cub": results_cub,
            "thrust": results_thrust}

if __name__ == '__main__':
    print(benchmark([1024, 32768, 1048576, 33554432]))