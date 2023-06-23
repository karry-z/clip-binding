#!/bin/bash
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem-per-gpu=64G
#SBATCH --partition gpu_short
#SBATCH --job-name=Investigate_CLIP_embeddings
#SBATCH --account=EMAT028344


DEBUG=false
# if to debug, uncomment the following line, then run the script directly with bash instead of sbatch
# DEBUG=true
# Change into working directory
cd /user/home/pu22650/clip-binding
#SBATCH --nodelist=bp1-gpu035
if [ $DEBUG = false ]; then
echo Running on host "$(hostname)"
echo Time is "$(date)"
start=$(date +%s.%N)
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"
fi

#Add the module you are using
module load lang/python/miniconda/3.9.7
bash ~/initConda.sh # don't submit the script in a env that has already been initialised
# Activate virtualenv
# conda activate ldm
source activate ldm
which python
if [ $DEBUG = false ]; then
nvidia-smi --query-gpu=gpu_name,memory.free --format=csv
fi
echo "---------------------------------------------"

# Execute code
if [ $DEBUG = false ]; then
srun -u python Investigate_CLIP_embeddings.py
else
echo "Debug code execute!"
# srun --pty -p gpu_short --gres=gpu:2 -A EMAT028344 python -m pdb accelerate_ddp.py
# https://code.visualstudio.com/docs/python/debugging
# launch.json "host" can be got from ```squeue -j jobid``` NODELIST(REASON), the jobid will be printed in the srun output.
srun --pty -p gpu_short --gres=gpu:1 --mem=64G -A EMAT028344 python -m debugpy --listen 0.0.0.0:5678 --wait-for-client accelerate_ddp.py
fi
echo "---------------------------------------------"
# Deactivate virtualenv
conda deactivate
echo End Time: $(date)
echo Time elapsed: $(echo "$(date +%s.%N) - $start" | bc) seconds
if [ $DEBUG = false ]; then
log_path=./job_management/slurm-${SLURM_JOBID}.out
sendmail pu22650@bristol.ac.uk < <( head $log_path && echo && tail $log_path )
fi
