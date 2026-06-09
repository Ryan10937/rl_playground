#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=rubiks-part
#SBATCH --time 10:00:00


#run script in env
source /home/ryan/rubiks/.env/bin/activate
python3 /home/ryan/rubiks/rl_playground/train_rl.py
