# -*- coding: utf-8 -*-

# -- stdlib --
from io import StringIO
from subprocess import PIPE, Popen
import logging
import platform

# -- third party --
import vulkan as vk

# -- own --

# -- code --
log = logging.getLogger('runner')


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
