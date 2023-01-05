# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark


# -- code --
class Fill(TaichiBenchmark):
    name = 'fill'
    archs = [ti.cuda, ti.vulkan]
    MB = 1024 * 256
    matrix = {
        'n': [64 * MB, 128 * MB,  512 * MB, 2048 * MB],
    }

    @property
    def repeats(self):
        MB = 1024 * 256
        if self.n < 1024 * MB:
            return 2000
        return 200

    def init(self, n):
        self.n = n
        self.a = ti.ndarray(ti.f32, n)
        self.run_iter = lambda : self.a.fill(0.125)

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        return {
            'GB/s': self.n * 4 / 2**30 / avg_time
        }
