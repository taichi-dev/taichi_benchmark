# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict
import os
import subprocess
import sys

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark

# -- code --
def get_file_size(file_path):
    try:
        file_size = os.path.getsize(file_path)
        return file_size
    except FileNotFoundError as e:
        print(f"Error: The file '{file_path}' does not exist.")
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise e

def git_clone(repo_url, dest_dir=None):
    try:
        command = ['git', 'clone', repo_url, '--recursive']

        if dest_dir:
            command.append(dest_dir)

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Git clone output:\n{result.stdout}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise e

def execute_bash_script(script_path, env):
    try:
        result = subprocess.run(['bash', script_path], capture_output=True, text=True, check=True, env=env)
        print(f"Script output:\n{result.stdout}")
    except FileNotFoundError as e:
        print(f"Error: The script file '{script_path}' does not exist.")
        raise e
    except subprocess.CalledProcessError as e:
        print(f"Error: The script exited with non-zero status ({e.returncode}).")
        print(f"Script output:\n{e.stdout}")
        print(f"Script error:\n{e.stderr}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise e


class CAPILibrarySize(TaichiBenchmark):
    name = 'c_api_library_size'
    archs = [ti.x64]

    def init(self):
        pass

    def run_iter(self):
        pass

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        taichi_path = os.path.join(curr_dir, 'taichi')

        # clone taichi
        if not os.path.exists(taichi_path):
            git_clone("https://github.com/taichi-dev/taichi.git", taichi_path)
        
        # compile for android
        env = os.environ.copy()
        env['TAICHI_REPO_DIR'] = taichi_path
        
        android_script_path = os.path.join(curr_dir, "build-taichi-android.sh")
        android_library_path = os.path.join("build-taichi-android-aarch64", "install", "c_api", "lib", "libtaichi_c_api.so")
        
        execute_bash_script(android_script_path, env)
        android_library_size = get_file_size(android_library_path)

        # compile for linux
        linux_script_path = os.path.join(curr_dir, "build-taichi-linux.sh")
        linux_library_path = os.path.join("build-taichi-linux", "install", "c_api", "lib", "libtaichi_c_api.so")
        
        execute_bash_script(linux_script_path, env)
        linux_library_size = get_file_size(linux_library_path)

        return {
            'linux': linux_library_size,
            'android': android_library_size,
        }
