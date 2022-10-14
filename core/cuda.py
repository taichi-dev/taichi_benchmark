# -*- coding: utf-8 -*-

# -- stdlib --
from contextlib import contextmanager
from subprocess import PIPE, Popen
import json
import logging
import os

# -- third party --
# -- own --

# -- code --
log = logging.getLogger('runner')


@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def cuda_compile_and_benchmark(workdir, source_name, output_binary_name, flags=[]):
    with pushd(workdir):
        # Compile
        p = Popen(['nvcc', '-O3', source_name, '-DJSON_OUTPUT',
                  '-o', output_binary_name] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot compile {}".format(output_binary_name))
        print("Successfully compiled {} into {}".format(
            source_name, output_binary_name))

        # Run Benchmark
        p = Popen(['./'+output_binary_name], stdout=PIPE)
        output, _ = p.communicate()
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
