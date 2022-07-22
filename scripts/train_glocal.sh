#!/bin/bash
#BSUB -n 8                      # 32 cores
#BSUB -W 24:00                   # 4-hour run-time
#BSUB -R "rusage[mem=8000, scratch=1000]"     # 4000 MB per core
# #BSUB -R "rusage[mem=8000, scratch=1000, ngpus_excl_p=1]"     # 4000 MB per core
#BSUB -J cilGlocal
#BSUB -o "/cluster/home/galtintas/cil/cilGlocal.out"
#BSUB -e "/cluster/home/galtintas/cil/cilGlocal.err"
#BSUB -N


source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.8.5 eth_proxy
# module load gcc/8.2.0 python_gpu/3.8.5 cuda/11.3.1/ eth_proxy
date; echo; echo
module list
# nvidia-smi; echo; echo

course="cil"
source "${SCRATCH}/${course}_env/bin/activate"
cd /cluster/home/$USER/cil/src


# python -m train --use-wandb True --experiment-dir $SCRATCH/cil \
#      --experiment-type optuna --algo glocal_k \
#     --n-trials 10  --enable-pruning True \
#     --NUM-WORKERS 8 --iter-p=5 --iter-f=5 \
#     --lr-fine=1 --lr-pre=0.1 \
#     --lambda-2=20 \
#     --lambda-s=0.006 \
#     --epoch-p=30 \
#     --epoch-f=60 \
#     --dot-scale=1 \
#     --gk-size=5 \
#     --train-size=0.9

#     --lr-fine 0.1 --lr-pre 0.1
    # --n-epochs 1 --epoch-p 1 --epoch-f 1 
date; echo; echo

python -m train --use-wandb True --experiment-dir $SCRATCH/cil \
     --experiment-type train --algo glocal_k \
    --NUM-WORKERS 8 \
    --n-hid 1000 \
    --n-dim 10 \
    --n-layers 2 \
    --gk-size 3 \
    --lambda-2 20 \
    --lambda-s 0.006 \
    --iter-p 5 \
    --iter-f 5 \
    --epoch-p 30 \
    --epoch-f 60 \
    --dot-scale 1 \
    --seed 42 \
    --lr-fine 1 --lr-pre 1
