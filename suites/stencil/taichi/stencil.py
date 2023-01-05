# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark


class Stencil(TaichiBenchmark):
    name = 'stencil'
    archs = [ti.cuda, ti.vulkan, ti.metal]
    matrix = {
        'N': [256, 1024, 4096],
    }

    def init(self, N):
        neighbours = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        x = ti.field(shape=(N, N), dtype=ti.float32)
        y = ti.field(shape=(N, N), dtype=ti.float32)

        @ti.kernel
        def init():
            for I in ti.grouped(x):
                if (I[0] == 0) or (I[0] == N - 1):
                    x[I] = 1.0
                else:
                    x[I] = 0.0

        @ti.kernel
        def jacobi_step():
            for I in ti.grouped(x):
                if (I[0] == 0) or (I[1] == 0) or (I[0] == N - 1) or (I[1] == N - 1):
                    y[I] = x[I]
                else:
                    p = 0.0
                    for stencil in ti.static(neighbours):
                        p += x[I + stencil]
                    y[I] = p * 0.25

            for I in ti.grouped(x):
                if (I[0] == 0) or (I[1] == 0) or (I[0] == N - 1) or (I[1] == N - 1):
                    x[I] = y[I]
                else:
                    p = 0.0
                    for stencil in ti.static(neighbours):
                        p += y[I + stencil]
                    x[I] = p * 0.25

        self.run_iter = jacobi_step
        self.N = N
        init()

    def get_metrics(self, avg_time):
        return {'fps': 1.0/avg_time, 'GB/s': 1e-9 * self.N * self.N * 4 * 2.0 / avg_time}
