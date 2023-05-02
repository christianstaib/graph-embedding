#!/bin/sh
#SBATCH --partition=fat
#SBATCH --time=240
#SBATCH --mem=128000
#SBATCH --output=out.txt
#SBATCH --error=err.txt

enroot start --root --rw -m $HOME/graph-embedding/GraphGPS:/GraphGPS -m $HOME/.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps:/GraphGPS/gps ubuntu+latest enroot_job.sh
