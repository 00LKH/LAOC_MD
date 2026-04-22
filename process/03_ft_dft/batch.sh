#!/bin/bash
#PBS -N mace_ft
#PBS -q full
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate mace_0316

# cd $PBS_O_WORKDIR
# python -u 00_vasprun2extxyz.py

cd $PBS_O_WORKDIR
python -u 01_data_replay.py

cd $PBS_O_WORKDIR
python -u 02_training.py
exit 0
