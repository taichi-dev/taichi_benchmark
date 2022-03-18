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
        p = Popen(['./'+output_binary_name], stdout=PIPE)
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
        print(results)
        return results


def benchmark():
    return {"cublas": compile_and_benchmark("cublas.cu", "saxpy_cublas", flags=['-lcublas']),
            "thrust": compile_and_benchmark("thrust.cu", "saxpy_thrust")}

if __name__ == '__main__':
    print(benchmark())
