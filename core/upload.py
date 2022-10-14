# -*- coding: utf-8 -*-

# -- stdlib --
from datetime import datetime
from typing import List
import json

# -- third party --
import requests

# -- own --
from .common import BenchmarkResult
from .machine import get_machine_info
from .taichi import get_taichi_version


# -- code --
def upload_benchmark_results(metrics: List[BenchmarkResult], auth: str):
    machine = get_machine_info()
    ts = str(datetime.now().astimezone())
    ver = get_taichi_version()

    payload = [{
        "time": ts,
        "name": m.name,
        "commit_id": ver,
        "tags": m.tags,
        "uploader": "",  # TODO
        "machine": machine,
        "value": m.value,
    } for m in metrics]

    resp = requests.post(
        url="https://benchmark.taichi-lang.cn/taichi_benchmark",
        data=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "Authorization": f'Bearer {auth}',
        },
    )
    print(resp.status_code)
