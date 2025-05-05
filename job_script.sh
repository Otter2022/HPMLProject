#!/bin/bash
#SBATCH --job-name=AlexnetPipe
#SBATCH --nodes=1                     
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=job-%j.out
#SBATCH --partition=gpu2v100

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Nodelist: " $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Using 1 task on this node."
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

module load cudatoolkit/12.5.0_555.42.02
module load anaconda3/2024.10-1

ENV_NAME="pipeline_env"
ENV_DIR="${SLURM_TMPDIR:-$HOME/tmp}/$ENV_NAME"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment at $ENV_DIR"
    conda create -p $ENV_DIR -y python=3.12
    source activate $ENV_DIR
    pip install torch torchvision fairscale scipy
else
    echo "Using existing conda environment at $ENV_DIR"
    source activate $ENV_DIR
    pip install torch torchvision fairscale scipy --upgrade
fi

echo "############### regular Script ###############"
python pipeline.py

echo "############### mixed percision Script ###############"
python pipeline_plus_mixedpercision.py