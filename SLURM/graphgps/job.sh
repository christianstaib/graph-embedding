#!/bin/sh
#SBATCH --time=240
#SBATCH --mem=128000
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --container-image ubuntu:latest
#SBATCH --container-name ubuntu
#SBATCH --container-mount-home
#SBATCH --container-writable

#-m $HOME/c-mount:/GraphGPS -m $HOME/.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps:/GraphGPS/gps
#enroot start --root --rw ubuntu+latest bash
pwd
ls -al
apt-get update && apt-get upgrade -y
apt-get install wget
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p
export PATH="$HOME/miniconda3/bin:$PATH"
cd ./c-mount/GraphGPS
conda activate $HOME/.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps
python main.py --cfg configs/GPS/ogbg-molpcba-GPS.yaml  wandb.use False
