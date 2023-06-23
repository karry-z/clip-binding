#!/bin/bash
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem-per-gpu=64G
#SBATCH --partition gpu_short
#SBATCH --job-name=clip_mlp_rel_5
#SBATCH --account=EMAT028344


# Change into working directory
cd /user/home/pu22650/clip-binding
log_path=./job_management/slurm-${SLURM_JOBID}.out

echo job name: ---"${SLURM_JOB_NAME}"---
echo host: ---"$(hostname)"---
echo Time: ---"$(date)"---
echo Directory: ---"$(pwd)"---
echo Slurm job ID: ---"${SLURM_JOBID}"---
echo This jobs runs on the following machines: ---"${SLURM_JOB_NODELIST}"---
echo ""

#Add the module you are using
module load lang/python/miniconda/3.9.7
bash ~/initConda.sh # don't submit the script in a env that has already been initialised
# Activate virtualenv
source activate ldm
which python
nvidia-smi --query-gpu=gpu_name,memory.free --format=csv
echo "--------------------LOG START-------------------------"

# Execute code
time \
srun -u python train_CLIP_mlp.py --model_name clip \
    --dataset rel \
    --epochs 5 \
    --save_dir /user/work/pu22650/clip-binding-out/clip_mlp_rel_5 \
    --save_model \
    --train_batch_size 32 \
    --eval_batch_size 64 

echo "--------------------LOG END-------------------------"
# Deactivate virtualenv
conda deactivate
sendmail pu22650@bristol.ac.uk < <( head $log_path && echo && tail $log_path )

