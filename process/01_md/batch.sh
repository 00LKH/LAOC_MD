#!/bin/bash
#PBS -N LAOC_md
#PBS -q full
#PBS -l select=1:ncpus=20:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate torchsim

cd $PBS_O_WORKDIR

python -u 00_mace_md_chain.py

exit 0
