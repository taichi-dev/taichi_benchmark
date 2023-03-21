# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Dict
import tempfile

# -- third party --
import taichi as ti
from .mpm import MPM

# -- code --
class MPMCompile(MPM):
    name = 'mpm_compile'
    
    @property
    def repeats(self):
        return 2

    def init(self, n_grid, dim):
        super().init(n_grid, dim)
        
        arch = ti.lang.impl.get_runtime().prog.config().arch
        self.mod = ti.aot.Module(arch) 

    def run_iter(self):
        for _ in range(self.steps):
            self.mod.add_kernel(self.mpm_substep)
    
    def iter_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.mod.save(temp_dir)

    def get_metrics(self, avg_time: float) -> Dict[str, float]:
        return {
            'fps': 1.0 / avg_time,
            'particles/s': self.n_particles * self.steps / avg_time
        }
