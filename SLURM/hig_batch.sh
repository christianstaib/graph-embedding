#!/bin/sh
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --gres=gpu:4
#SBATCH --time=30
#SBATCH --container-image=./ubuntu+latest.sqsh
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,./HIG/HIG/:/HIG,graph-embedding/hig_examples:/hig_examples
#SBATCH --container-writable
#SBATCH --container-remap-root
#SBATCH --output=R_HIG-%x-%j.out
#SBATCH --error=R_HIG-%x-%j.err

apt-get update && apt-get upgrade -y
cd /root
apt-get install build-essential
apt-get install wget -y
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -f -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda update -y conda
conda init bash
conda create -n hig  python=3.10
conda activate hig

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

conda install pytorch-lightning yacs torchmetrics
conda install cython
pip install performer-pytorch
conda install tensorboardX
pip install ogb
pip install wandb
conda install networkx
conda install bash
conda install numpy=1.20.3
conda install pandas=1.2.5
conda install scikit-learn=0.24.2
pip install libauc
conda clean --all

cd /HIG/Graphormer_with_HIG
cp -r /hig_examples/* ./examples/
sh ./examples/ogb-lsc/lsc-pcba.sh
sh ./examples/ogb/finetune_dropnode.sh
sh ./examples/ogb/finetune_kl.sh
