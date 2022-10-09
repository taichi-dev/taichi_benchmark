from contextlib import contextmanager
import os
import time
import json
import platform
import copy
import taichi as ti

from subprocess import Popen, PIPE

from taichi.lang import cpu, cuda, metal, opengl, vulkan, dx11
from taichi._lib import core as _ti_core


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


def get_taichi_version():
    return (f'{_ti_core.get_version_major()}.{_ti_core.get_version_minor()}.{_ti_core.get_version_patch()}', _ti_core.get_commit_hash())

def run_taichi_benchmark(func, arch, **kwargs):
    repeats = kwargs.get('repeats')
    ti.init(arch=arch, device_memory_GB=4,
            offline_cache=False)
    run_init, run_iter, metrics = func(**kwargs)
    run_init()

    st = time.perf_counter()
    for _ in range(repeats):
        run_iter()
    ti.sync()
    et = time.perf_counter()
    avg_time = (et - st) / repeats
    return avg_time, metrics(avg_time)
    

def benchmark(test_name, archs=[cuda, vulkan, metal, opengl, dx11], **options):
    def is_target_platform(arch):
        cur_system = platform.system()
        # For MacOS
        if cur_system == 'Darwin':
            if arch in (cuda, opengl, dx11):
                return False
        # For Linux
        if cur_system == 'Linux':
            if arch in (metal, dx11):
                return False
        # For Windows 
        # TODO TEST Windows backends
        if cur_system == 'Windows':
            if arch in (opengl, metal):
                return False
        return True

    def decorator(func):
        def run_benchmark(**kwargs):
            results = []
            for arch in archs:
                if not is_target_platform(arch):
                    continue
                config = copy.deepcopy(kwargs)
                config['taichi_version'] = get_taichi_version()
                config['arch'] = arch
                avg_time, metrics = run_taichi_benchmark(func, arch, **kwargs)
                results.append(
                    {'test_name' : test_name, 'test_config' : config, 'wall_time' : avg_time*1000.0, 'metrics' : metrics})
            return results
        return run_benchmark

    return decorator
