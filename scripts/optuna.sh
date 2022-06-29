#!/bin/bash
#BSUB -n 4                      # 32 cores
#BSUB -W 72:00                   # 4-hour run-time
#BSUB -R "rusage[mem=2000, scratch=1000]"     # 4000 MB per core
#BSUB -J cilOptuna
#BSUB -o "/cluster/home/galtintas/cil/cilOptuna.out"
#BSUB -e "/cluster/home/galtintas/cil/cilOptuna.err"
#BSUB -N


source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.8.5 eth_proxy
# module load gcc/8.2.0 python_gpu/3.8.5 cuda/11.3.1 eth_proxy
date; echo; echo
module list
# nvidia-smi; echo; echo

course="cil"
source "${SCRATCH}/${course}_env/bin/activate"
# cd /cluster/home/$USER/cil/models/ncf/src
cd /cluster/home/$USER/cil/

python baselines_optuna.py --use-wandb=False --algo=svdpp --n-trials=50 --n-jobs=4 --home="/cluster/home/galtintas/"
# python baselines_optuna.py --use-wandb=False --algo=svdpp --n-trials=50

### ! BSUB -R "rusage[mem=2000, scratch=10000, ngpus_excl_p=1]"     # 4000 MB per core
### ! BSUB -R "select[gpu_model0==GeForceGTX1080]" 
date; echo; echo
