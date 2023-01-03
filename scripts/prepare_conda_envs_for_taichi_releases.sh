#!/bin/bash

ENV_PREFIX="taichi_perfmon_releases"
TI_VERSIONS=(1.0.0 1.0.1 1.0.2 1.0.3 1.0.4 1.1.0 1.1.2 1.1.3 1.2.0 1.2.1 1.2.2 1.3.0 master)
PY_VERSION="3.9"
CONDA_ENVS=`conda env list`
for TI_VER in ${TI_VERSIONS[@]}
do
ENV_NAME="${ENV_PREFIX}_${TI_VER}"
ENV_AVAIL=`echo ${CONDA_ENVS}|grep ${ENV_NAME}`
if [ -z "$ENV_AVAIL" ]; then
echo "[Taicho] Taichi release environment ${ENV_NAME} not available yet. Create it!"
echo "[Conda] Creating conda env ${ENV_NAME} with Python ${PY_VERSION}..."
conda create -y -n ${ENV_NAME} python=${PY_VERSION}
echo "[Pip] Installing taichi ${TI_VER}"
CONDA_PATH=`conda env list |grep ${ENV_NAME}|awk '{print $NF}'`
echo $PIP_PATH
fi
CONDA_PATH=`conda env list |grep ${ENV_NAME}|awk '{print $NF}'`
PIP_PATH="${CONDA_PATH}/bin/pip"
$PIP_PATH install taichi==${TI_VER} requests vulkan
echo "[Taichi] Verifying Taichi installation ver ${TI_VER}"
PY_PATH="${CONDA_PATH}/bin/python"
TI_INIT_STR=`$PY_PATH -c "import taichi"`
echo $TI_INIT_STR
if [[ $TI_INIT_STR =~ "[Taichi] version ${TI_VER}" ]]; then
    echo "====================OK!====================="
else
    echo "Error in installing Taichi ${TI_VER}! Aborted!"
    exit -1
fi
done
