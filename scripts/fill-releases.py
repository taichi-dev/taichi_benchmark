# -*- coding: utf-8 -*-

# -- stdlib --
from datetime import datetime
import re
import os

# -- third party --
import requests

# -- own --

# -- code --
# Replace with your GitHub access token
GITHUB_API_URL = "https://api.github.com"
GITHUB_REPO = "taichi-dev/taichi"
POSTGREST_ENDPOINT = "https://benchmark.taichi-lang.cn/releases"

headers = {
    "Accept": "application/vnd.github+json",
    'X-GitHub-Api-Version': '2022-11-28',
    'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}',
}

# Function to convert version string to numerical version number
def version_to_vnum(version):
    nums = re.findall(r'\d+', version)
    return int(''.join([f"{num:0>2}" for num in nums]))

# Function to fetch and process tags
def process_tags():
    url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/tags"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching tags: {response.text}")
        return

    tags = response.json()
    for tag in tags:
        version = tag['name']
        if re.match(r'^v\d+\.\d+\.\d+$', version):
            vnum = version_to_vnum(version)
            commit_url = tag['commit']['url']
            commit_response = requests.get(commit_url, headers=headers)

            if commit_response.status_code != 200:
                print(f"Error fetching commit: {commit_response.text}")
                continue

            commit = commit_response.json()
            commit_id = commit['sha']
            commit_time = datetime.fromisoformat(commit['commit']['committer']['date'].replace('Z', '+00:00'))

            yield {
                "version": version,
                "vnum": vnum,
                "commit_id": commit_id,
                "commit_time": commit_time.isoformat()
            }

# Post the data to the PostgREST endpoint
def post_to_postgrest(data):
    headers = {
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
        "Authorization": f"Bearer {os.environ['BENCHMARK_TAICHI_CN_POSTGREST_TOKEN']}",
    }
    response = requests.post(POSTGREST_ENDPOINT, json=data, headers=headers)
    if response.status_code != 201:
        print(f"Error posting data: {response.text}")


if __name__ == "__main__":
    for data in process_tags():
        post_to_postgrest(data)
        print(f"Posted data: {data}")
