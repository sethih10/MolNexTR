#!/bin/bash
#SBATCH --job-name=train_script          # Job name
#SBATCH --output=exps/train_%j.log       # Output log file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --nodes=1                        # Number of nodes (matches NUM_NODES)
#SBATCH --gpus=8                         # GPUs per node (matches NUM_GPUS_PER_NODE)
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec


# Load necessary modules

# Activate your environment
source /scratch/phys/sin/sethih1/venv/MolNexTR_env/bin/activate

# Set environment variables
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export BATCH_SIZE=64
export ACCUM_STEP=1


# Run the train.sh script
sh ./exps/train.sh
