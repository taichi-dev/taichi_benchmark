#!/bin/bash

ENV_PREFIX="taichi_perfmon_releases"
TI_VERSIONS=(1.0.0 1.0.4 1.1.3 1.2.2 1.3.0)
WORKDIR=`dirname $0`/..
WORKDIR=`readlink -f $WORKDIR`
TIMESTAMP=`date -I seconds`
LOGDIR=$WORKDIR/perfmon_log_${TIMESTAMP}
mkdir -p $LOGDIR
pushd $WORKDIR
for TI_VER in ${TI_VERSIONS[@]}
do
ENV_NAME="${ENV_PREFIX}_${TI_VER}"
CONDA_PATH=`conda env list |grep ${ENV_NAME}|awk '{print $2}'`
PY_EXEC="${CONDA_PATH}/bin/python"
$PY_EXEC run.py --save ${LOGDIR}/benchmark_v${TI_VER}.log
done
popd