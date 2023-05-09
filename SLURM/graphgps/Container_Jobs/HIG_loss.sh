#!/bin/sh
#SBATCH --partition=fat
#SBATCH --time=480
#SBATCH --container-image=./ubuntu+latest.sqsh
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,./graph-embedding:/GraphGPS
#SBATCH --container-writable
#SBATCH --container-remap-root
#SBATCH --output=OUT_loss.txt
#SBATCH --error=ERR_loss.txt

apt-get update && apt-get upgrade -y
cd /root
apt-get install wget -y
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
pwd
ls -al
bash ~/miniconda.sh -b -f -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda update -y conda
conda init bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb
pip install networkx
conda clean --all
cd /GraphGPS/GraphGPS

python main.py --cfg configs/GPS/ogbg-molhiv-GPS+HIG_loss.yaml  wandb.use False
