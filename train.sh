#!/bin/bash
#SBATCH --job-name=ocr_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=registered@email.address
#SBATCH --time=00:15:00 # Using the default queue, jobs cannot exceed 24 hours
# Or
# #SBATCH --qos=long 
# #SBATCH --time=2-00:00:00 # i.e. 2 days, you can go up to 7 with long QOS
#SBATCH --output=logs/ocr_train_%j.log
#SBATCH --error=logs/ocr_train_%j.err
#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Enroot container configurations
#SBATCH --container-image=/home/shared/air/enroot-images/pytorch_n_friends.sqsh
#SBATCH --container-mount-home
#SBATCH --container-mounts=/home/shared:/home/shared:ro

# Unload modules that could interfere
module purge

# 1. Navigate to the working directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# 2. Run the training script (sourcing the venv and running python in the same container step)
echo "Starting training job in $PWD..."
srun bash -c "cd $SLURM_SUBMIT_DIR && source .venv/bin/activate && python train_ocr.py --loader superfast --batch-size 256"