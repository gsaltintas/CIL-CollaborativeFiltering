#!/bin/bash
#BSUB -n 8                     
#BSUB -W 4:00                   
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

python -m train --experiment-dir $SCRATCH/cil \
     --experiment-type train --algo glocal_k \
    --NUM-WORKERS 8  --n-hid 1000  --n-dim 5  --n-layers 2  --gk-size 5 \
    --lambda-2 20  --lambda-s 0.006  --iter-p 5  --iter-f 5 \
     --epoch-p 30 --epoch-f 80 --dot-scale 1 --seed 1234 \
    --lr-fine 0.1 --lr-pre 0.1 --optimizer lbfgs --lr-scheduler none
