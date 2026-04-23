#!/bin/bash
#PBS -N LAOC_tag
#PBS -q full
#PBS -l select=1:ncpus=20:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate mace

cd $PBS_O_WORKDIR
python -u 00_Cl.py

cd $PBS_O_WORKDIR
python -u 01_Li.py

exit 0
