#!/bin/sh
#SBATCH --partition=fat
#SBATCH --time=240
#SBATCH --mem=128000
#SBATCH --container-image=./ubuntu+latest
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,./graph-embedding:/GraphGPS,./.local/share/enroot/ubuntu+latest/root/miniconda3/envs/gps:/GraphGPS/gps
#SBATCH --output=out.txt
#SBATCH --error=err.txt

apt update && apt upgrade -y
cd /root
apt install wget -y
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
pwd
ls -al
bash ~/miniconda.sh -b -f -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda update -y conda
conda init bash
cd /GraphGPS
conda activate ./gps
python main.py --cfg configs/GPS/ogbg-molpcba-GPS.yaml  wandb.use False
