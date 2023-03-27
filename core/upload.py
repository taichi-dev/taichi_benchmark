# -*- coding: utf-8 -*-

# -- stdlib --
from datetime import datetime
from typing import List, Dict
import json
import logging
from pathlib import Path
import subprocess

# -- third party --
import requests

# -- own --
from .common import BenchmarkResult
from .machine import get_machine_info
from .taichi import get_taichi_version


# -- code --
log = logging.getLogger(__name__)


def find_taichi_root():
    root = Path('/')
    p = Path(__file__).resolve().parent
    while p != root:
        if (p / 'setup.py').exists():
            return p
        p = p.parent


def get_commit_time(commit_id: str) -> datetime:
    root = find_taichi_root()
    cmd = f'cd {root} && git show -s --format=%ct {commit_id}'
    output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    return datetime.fromtimestamp(int(output))


def upload_results(metrics: List[BenchmarkResult], auth: str, tags: Dict[str, str] = {}):
    machine = get_machine_info()
    ver = get_taichi_version()
    ts = str(get_commit_time(ver).astimezone())

    payload = [{
        "time": ts,
        "name": m.name,
        "commit_id": ver,
        "tags": {**tags, **m.tags},
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
