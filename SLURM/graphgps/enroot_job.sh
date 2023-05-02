#!/bin/sh
apt-get update && apt-get upgrade -y
apt-get install wget
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p
export PATH="$HOME/miniconda3/bin:$PATH"
cd ./graph-embedding/GraphGPS
conda activate $HOME/.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps
python main.py --cfg configs/GPS/ogbg-molpcba-GPS.yaml  wandb.use False
