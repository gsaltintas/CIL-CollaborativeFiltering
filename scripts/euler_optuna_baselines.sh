#!/bin/bash
#BSUB -n 4                      
#BSUB -W 24:00                   
#BSUB -R "rusage[mem=2000, scratch=1000]"     
#BSUB -J cilOptunaBaselines
#BSUB -N


source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.8.5 eth_proxy
date; echo; echo
module list

course="meowtrix-purrdiction"
source "${SCRATCH}/${course}_env/bin/activate"
cd /cluster/home/$USER/$course/src
algo="svdpp"  # svd

python -m train --experiment-type optuna --use-wandb=False --algo=$algo \
    --experiment-dir $SCRATCH/baselines --n-trials=50 --n-jobs=1 \
    --seed=1234

