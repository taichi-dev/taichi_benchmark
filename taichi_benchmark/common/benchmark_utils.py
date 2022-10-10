from contextlib import contextmanager
import os
import time
import json
import platform
import copy
from io import StringIO
from subprocess import Popen, PIPE

import taichi as ti
from taichi.lang import cpu, cuda, metal, opengl, vulkan, dx11
from taichi._lib import core as _ti_core

import vulkan as vk


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


def run_taichi_benchmark(func, arch, repeats, **kwargs):
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


def get_cpu_name_linux():
    p = Popen(['cat', '/proc/cpuinfo'], stdout=PIPE)
    output, err = p.communicate()
    strio = StringIO(output.decode('utf-8'))
    for line_raw in strio:
        line = line_raw.strip()
        if line.startswith('model name'):
            return line.split(':')[1].strip()


def get_gpu_name_nvidia_smi():
    p = Popen(['nvidia-smi', '-q'], stdout=PIPE)
    output, err = p.communicate()
    if err != None:
        return None
    strio = StringIO(output.decode('utf-8'))
    for line_raw in strio:
        line = line_raw.strip()
        if line.startswith('Product Name'):
            return line.split(':')[1].strip()


def get_vulkan_device_name():
    application_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext=None,
        pApplicationName="Taichi Benchmark",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    )

    instance_create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=0,
        pApplicationInfo=application_info,
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )
    instance = vk.vkCreateInstance(instance_create_info, None)
    vk_devices = vk.vkEnumeratePhysicalDevices(instance)
    for device in vk_devices:
        device_name = vk.vkGetPhysicalDeviceProperties(device).deviceName
        if device_name.startswith('llvmpipe'):
            continue
        return device_name
    return None


def get_gpu_name():
    gpu_name = get_gpu_name_nvidia_smi()
    if gpu_name == None:
       gpu_name = get_vulkan_device_name()
    return gpu_name


def get_processor_names():
    cur_system = platform.system()
    if cur_system == "Linux":
        return {'CPU': get_cpu_name_linux(), 'GPU': get_gpu_name()}
    else:
        raise NotImplementedError


def get_os_name():
    cur_uname = platform.uname()
    return f'{cur_uname.system} {cur_uname.release}'

def get_machine_info():
    machine_info = get_processor_names()
    machine_info['os'] = get_os_name()
    return machine_info

def benchmark(test_name, archs=[cuda, vulkan, metal, opengl, dx11], default_repeats=20, **options):
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
            repeats = default_repeats
            # Override the repeats present in options
            if kwargs.get('repeats') != None:
                repeats = kwargs['repeats']
                kwargs.pop('repeats', None)

            for arch in archs:
                if not is_target_platform(arch):
                    continue
                config = copy.deepcopy(kwargs)
                config['taichi_version'] = get_taichi_version()
                config['arch'] = arch
                avg_time, metrics = run_taichi_benchmark(
                    func, arch, repeats, **kwargs)
                metrics['wall_time'] = avg_time
                results.append(
                    {'test_name': test_name, 'test_config': config, 'metrics': metrics})
            return results
        return run_benchmark

    return decorator
