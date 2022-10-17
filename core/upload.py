# -*- coding: utf-8 -*-

# -- stdlib --
from datetime import datetime
from typing import List
import json
import logging

# -- third party --
import requests

# -- own --
from .common import BenchmarkResult
from .machine import get_machine_info
from .taichi import get_taichi_version


# -- code --
log = logging.getLogger(__name__)


def upload_results(metrics: List[BenchmarkResult], auth: str):
    machine = get_machine_info()
    ts = str(datetime.now().astimezone())
    ver = get_taichi_version()

    payload = [{
        "time": ts,
        "name": m.name,
        "commit_id": ver,
        "tags": m.tags,
        "uploader": "",  # TODO: end user uploading
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
    resp.raise_for_status()
    log.info('Upload successful')
