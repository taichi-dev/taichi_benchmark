# -*- coding: utf-8 -*-
from __future__ import annotations

# -- stdlib --
from typing import Any, ClassVar, Dict, List
import logging
import platform
import time

# -- third party --
from taichi._lib import core as _ti_core
from taichi.lang import cuda, dx11, metal, opengl, vulkan
import taichi as ti

# -- own --
from .common import BenchmarkResult, iter_matrix


# -- code --
log = logging.getLogger('runner')


def get_taichi_version():
    return _ti_core.get_commit_hash()


class TaichiBenchmarkRunner:

    @staticmethod
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

    def run(self, bm: TaichiBenchmark) -> List[BenchmarkResult]:
        results = []
        for arch in bm.archs:
            if not self.is_target_platform(arch):
                continue

            for cfg in iter_matrix(bm.matrix):
                try:
                    if not bm.check_configuration(**cfg):
                        continue

                    tags = {**bm.tags, **cfg, 'arch': arch.name, 'impl': 'Taichi'}
                    ti.init(arch=arch, device_memory_GB=bm.memory_gb, offline_cache=False)
                    bm.init(**cfg)
                    repeats = bm.repeats
                    log.info('Testing %s with configuration %s, repeats %s times', bm.name, tags, repeats)

                    bm.run_iter()

                    st = time.perf_counter()
                    for _ in range(repeats):
                        bm.run_iter()
                    ti.sync()
                    et = time.perf_counter()
                    avg_time = (et - st) / repeats

                    metrics = bm.get_metrics(avg_time)
                    metrics['wall_time'] = avg_time

                    for k, v in metrics.items():
                        results.append(
                            BenchmarkResult(name=f'{bm.name}:{k}', tags=tags, value=v)
                        )
                except Exception:
                    log.exception(f'Failed to run {bm.name} on {arch}')

        return results


class TaichiBenchmark:
    name: str
    memory_gb: float = 4
    archs: ClassVar[List[Any]] = [cuda, vulkan, metal, opengl, dx11]
    matrix: ClassVar[Dict[str, List[Any]]] = {}
    tags: ClassVar[Dict[str, Any]] = {}

    repeats: int = 20
    runner = TaichiBenchmarkRunner

    def check_configuration(self, **_):
        return True

    def init(self, **_):
        raise NotImplementedError

    def run_iter(self):
        raise NotImplementedError

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        raise NotImplementedError
