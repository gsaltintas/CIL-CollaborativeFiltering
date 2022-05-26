#!/bin/bash

course="cil"
username="galtintas"
echo "================"
echo "Current username: ${username}. Please change it if you are a different user"
echo "================"

SCRATCH="/cluster/scratch/${username}"
RUNTIME="06:00"
N_GPUS=1
MEM=4096
N_CORES=4
bash Jupyter-on-Euler-or-Leonhard-Open/start_jupyter_nb.sh --username "${username}" --numcores $N_CORES --runtime $RUNTIME \
        --memory $MEM --numgpu $N_GPUS \
        --workdir "${course}" \
	    --environment "${SCRATCH}/${course}_env"
