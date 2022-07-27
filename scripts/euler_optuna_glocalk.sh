#!/bin/bash
#BSUB -n 8                     
#BSUB -W 4:00                   # 4-hour run-time
#BSUB -R "rusage[mem=8000, scratch=1000]"     
#BSUB -J cilGlocal
#BSUB -N


source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.8.5 eth_proxy
date; echo; echo
module list

course="meowtrix-purrdiction"
source "${SCRATCH}/${course}_env/bin/activate"
cd /cluster/home/$USER/$course/src

python -m train --use-wandb False --experiment-dir  $SCRATCH/cil \
     --experiment-type optuna --algo glocal_k \
    --n-trials 20  --enable-pruning True \
    --NUM-WORKERS 1 --iter-p=5 --iter-f=5 \
    --lr-fine=0.1 --lr-pre=0.1 \
    --lambda-2=20 \
    --lambda-s=0.006 \
    --epoch-p=30 \
    --epoch-f=80 \
    --dot-scale=1 \
    --optimizer lbfgs \
    --scheduler none \
    --use-storage=False \
    --study-name test