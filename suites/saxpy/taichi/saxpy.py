# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict

# -- third party --
import taichi as ti

# -- own --
from core.taichi import TaichiBenchmark


class SAXPY(TaichiBenchmark):
    name = 'nested-saxpy'
    # Temp disable opengl,
    # breaks after updating nvidia driver
    # archs = [ti.cuda, ti.opengl, ti.vulkan, ti.metal]
    archs = [ti.cuda, ti.vulkan, ti.metal]
    matrix = {
        'n': [256, 1024, 2048, 4096],
        'len_coeff': [1, 32, 64, 128, 256]
    }

    @property
    def repeats(self):
        if 2 ** self.len_coeff < 256 and self.n < 2048:
            return 2000
        return 500

    def init(self, n, len_coeff):
        self.n = n
        self.len_coeff = len_coeff

        x = ti.field(shape=(n, n), dtype=ti.float32)
        y = ti.field(shape=(n, n), dtype=ti.float32)

        coeff = [2 * i * i * i / 3.1415926535 for i in range(1, len_coeff + 1)]

        @ti.kernel
        def init(x: ti.template()):
            for i in ti.grouped(x):
                x[i] = ti.random(ti.float32) * 64.0

        @ti.kernel
        def saxpy_kernel(x: ti.template(), y: ti.template()):
            for i in ti.grouped(y):
                # Statically unroll
                z_c = x[i]
                for c in ti.static(coeff):
                    z_c = c * z_c + y[i]
                y[i] = z_c

        init(x)
        init(y)

        self.run_iter = lambda: saxpy_kernel(x, y)

    def get_metrics(self, avg_time):
        n = self.n
        gflops = 1e-9 * self.len_coeff * 2 * n * n / avg_time
        gbs =  1e-9 * n * n * 4 * 3 / avg_time
        return {'gflops': gflops, 'gbs': gbs}
