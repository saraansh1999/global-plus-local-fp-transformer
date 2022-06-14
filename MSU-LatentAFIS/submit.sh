#!/bin/bash
#SBATCH -A $USER
#SBATCH -n 1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=15
#SBATCH --nodelist=node02

#module add cuda/8.0
#module add cudnn/7-cuda-8.0

export CUDA_VISIBLE_DEVICES=3
python extraction/pipeline.py --gpu 0 \
--mode gt --patch_types 1 --data_format all_in_one \
--input_dir $1 \
--minu_from $4 --minu_dir $2 \
--output_dir $3 \
> $5 2>&1 
