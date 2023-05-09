#!/bin/sh
#SBATCH --partition=fat
#SBATCH --time=240
#SBATCH --mem=128000
#SBATCH --output=out.txt
#SBATCH --error=err.txt

enroot start --root --rw -m $HOME/graph-embedding/GraphGPS:/GraphGPS -m $HOME/conda_gps:/GraphGPS/gps -m $HOME/graph-embedding/SLURM/graphgps/enroot_job.sh:/root/enroot_job.sh ubuntu+latest sh /root/enroot_job.sh
