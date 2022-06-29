#!/bin/bash
#BSUB -n 32                      # 32 cores
#BSUB -W 4:00                   # 4-hour run-time
#!!BSUB -R "rusage[mem=2000, scratch=10000, ngpus_excl_p=1]"     # 4000 MB per core
#BSUB -J cilTraining
#BSUB -o "/cluster/home/galtintas/cil/cilTraining.out"
#BSUB -e "/cluster/home/galtintas/cil/cilTraining.err"
#BSUB -N

echo Logging gradients

source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.8.5 eth_proxy
# module load gcc/8.2.0 python_gpu/3.8.5 cuda/11.3.1 eth_proxy
date; echo; echo
module list
# nvidia-smi; echo; echo

course="cil"
source "${SCRATCH}/${course}_env/bin/activate"
cd /cluster/home/$USER/cil/models/ncf/src

# echo "Running value_of_info/value_experiment.py"
python -m train

### ! BSUB -R "rusage[mem=2000, scratch=10000, ngpus_excl_p=1]"     # 4000 MB per core
### ! BSUB -R "select[gpu_model0==GeForceGTX1080]" 
date; echo; echo
