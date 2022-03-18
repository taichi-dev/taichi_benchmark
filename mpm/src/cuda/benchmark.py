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

def compile_and_benchmark(output_binary_name, flags=[]):
    workdir = os.path.dirname(os.path.abspath(__file__))
    with pushd(workdir):
        # Compile
        p = Popen(['cmake', '-S', '.', '-B', 'build'] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot generate cmake{}".format(output_binary_name))

        p = Popen(['cmake', '--build', 'build'] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot compile {}".format(output_binary_name))
        print("Successfully compiled MPM")

        # Run Benchmark
        results = []
        ## TODO, verify these data points
        configs = [(16, 20), (32, 20), (48, 20), (64, 20), 
                   (80, 20), (96, 20), (112, 20), (128, 20), 
                   (144, 32), (160, 32), (176, 32), (192, 32), 
                   (208, 32), (224, 32), (240, 32), (256, 32)]
        # Taichi does not allow power of two
        #configs = [(16, 20), (32, 20), (64, 20), 
        #           (128, 20), (256, 32)]
        for n_grid, step in configs:
            print("running", output_binary_name, "n_grid", n_grid, "step", step)
            argv = ["{}".format(n_grid), "{}".format(step)]
            results += run_binary(output_binary_name, argv)
        print("{} test finished.".format(output_binary_name))
        return results

def benchmark():
    return {"cuda_baseline": compile_and_benchmark('build/MPM')}

if __name__ == '__main__':
    print(benchmark())
