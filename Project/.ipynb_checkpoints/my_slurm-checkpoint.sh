#!/bin/bash

#SBATCH -A ds_6050
#SBATCH -p gpu                               # Specify the GPU partition
#SBATCH --gres=gpu:rtx3090:2                 # Request 2 RTX 3090 GPUs
#SBATCH -c 4                                 # Request 2 cores per GPU
#SBATCH -t 00:30:00                          # Set a time limit of 30 minutes
#SBATCH -J data_sys                          # Job name
#SBATCH -o data_sys-%A.out                   # Standard output file
#SBATCH -e data_sys-%A.err                   # Standard error file

# ijob -A ds_6050 -p gpu --nodes=4 --gres=gpu:rtx3090:2 -c 4 -t 00:30:00 --mem=32G
module purge
# Load modules
module load apptainer/1.2.2 pytorch/2.0.1 java/11 gcc/11.4.0 openmpi/4.1.4 python/3.11.4 spark/3.4.1

# apptainer exec --nv $CONTAINERDIR/pytorch-2.0.1.sif pip install pyspark
# apptainer exec --nv $CONTAINERDIR/pytorch-2.0.1.sif pip install spark-nlp
# apptainer exec --nv $CONTAINERDIR/pytorch-2.0.1.sif pip install findspark

apptainer exec --nv $CONTAINERDIR/pytorch-2.0.1.sif Documents/MSDS/DS5110/Project/sdl_debugger.py 