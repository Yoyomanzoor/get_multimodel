#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=6:00:00
#SBATCH --account=bioinf593f25_class
#SBATCH --partition=gpu
#SBATCH --mail-user=smanzoor@umich.edu
#SBATCH --job-name=run_diffusion-transformer
#SBATCH --get-user-env
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL

module load Bioinformatics
module load tabix
module load bedtools2
module load sratoolkit

python run_diffusion-transformer.py
