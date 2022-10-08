import glob
import multiprocessing
import os
import platform
import shutil
import sys
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

from setuptools import find_packages
from skbuild import setup
from skbuild.command.egg_info import egg_info

root_dir = os.path.dirname(os.path.abspath(__file__))

def get_version():
    if os.getenv("RELEASE_VERSION"):
        version = os.environ["RELEASE_VERSION"]
    else:
        version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
        with open(version_file, 'r') as f:
            version = f.read().strip()
    return version.lstrip("v")

project_name = "taichi_benchmark"
version = get_version()

setup(name=project_name,
    #   packages=packages,
    #   package_dir={"": package_dir},
      version=version,
      description='The Benchmark Suite for Taichi Programming Language',
      author='Taichi developers',
      author_email='haidonglan@taichi.graphics',
      url='https://github.com/taichi-dev/taichi_benchmark',
      python_requires=">=3.6,<3.11",
      install_requires=[
      ],
    #   data_files=[(os.path.join('_lib', 'runtime'), data_files)],
      keywords=['benchmark', 'taichi'],
      license=
      'Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
    #   include_package_data=True,
    #   entry_points={
    #       'console_scripts': [
    #           'ti=taichi._main:main',
    #       ],
    #   },
    #   classifiers=classifiers,
    #   cmake_args=get_cmake_args(),
    #   cmake_process_manifest_hook=exclude_paths,
    #   cmdclass={
    #       'egg_info': EggInfo,
    #       'clean': Clean
    #   },
    #   has_ext_modules=lambda: True
      )