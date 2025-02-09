#!/bin/bash
source /Users/filiproch/miniconda3/etc/profile.d/conda.sh
conda activate my-lammps-env
mpirun -n 1 python run.py
