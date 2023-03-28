# -*- coding: utf-8 -*-

# -- stdlib --
from dataclasses import dataclass
from typing import Any, Dict, List
import itertools

# -- third party --
# -- own --

# -- code --
@dataclass
class BenchmarkResult:
    name: str
    tags: Dict[str, str]
    value: float


def iter_matrix(matrix: Dict[str, List[Any]]):
    if not matrix:
        yield {}

    kl = list(matrix.keys())
    for vl in itertools.product(*matrix.values()):
        yield dict(zip(kl, vl))
