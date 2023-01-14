#!/bin/bash
#SBATCH --time=120
#SBATCH --ntasks-per-node=40
#SBATCH --mem=64000
#SBATCH --output=out.txt
#SBATCH --error=err.txt

enroot start --root --rw -m $HOME/python_job:/python_job tensorflow
cd python_job
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
