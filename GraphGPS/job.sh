#!/bin/sh
#SBATCH --time=240
#SBATCH --mem=128000
#SBATCH --output=out.txt
#SBATCH --error=err.txt

#-m $HOME/c-mount:/GraphGPS -m $HOME/.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps:/GraphGPS/gps
enroot start --root --rw ubuntu+latest bash
pwd
ls -al
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
export PATH="$HOME/miniconda3/bin:$PATH"
cd ./c-mount/GraphGPS
conda activate $HOME/.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps
python main.py --cfg configs/GPS/ogbg-molpcba-GPS.yaml  wandb.use False