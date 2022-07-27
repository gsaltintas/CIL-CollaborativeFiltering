#!/bin/bash
#BSUB -n 4                      
#BSUB -W 24:00                   
#BSUB -R "rusage[mem=2000, scratch=1000]"     
#BSUB -J cilBaselines
#BSUB -N


source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.8.5 eth_proxy
date; echo; echo
module list

course="meowtrix-purrdiction"
source "${SCRATCH}/${course}_env/bin/activate"
cd /cluster/home/$USER/$course/src

# svdpp
python -m train --experiment-type train  --use-wandb=False --algo=svdpp --n-factors=200 --n-epochs=60 --lr-all=0.005 --reg-all=0.1 --seed=42

# svd
python -m train  --experiment-type=train --use-wandb=False --algo=svd --n-factors=66 --biased=True --lr-all=0.014034216851890047 --reg-all=0.0904990847308604 --init-mean=0.9995295507254714 --init-std-dev=0.3654283107617773 --seed=42