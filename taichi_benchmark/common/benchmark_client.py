import requests
import json
from datetime import datetime
from taichi_benchmark.common import get_machine_info

def upload_benchmark_results(results, auth):
    machine = get_machine_info()
    for r in results:
        resp = requests.post(
            url="https://benchmark.taichi-lang.cn/taichi_benchmark",
            data=json.dumps({
                "time": str(datetime.now().astimezone()),
                "name": r["test_name"],
                "commit": {
                    "id": r["test_config"]["taichi_version"][1],
                    "version": r["test_config"]["taichi_version"][0],
                },
                "tags": {"particles": 512},
                "uploader": "",
                "machine": machine,
                "value": r["metrics"]["wall_time"],
            }),
            headers={
                "Content-Type": "application/json",
                "Authorization": auth,
            },
        ) 
        print(resp.status_code)
        print(resp.json())