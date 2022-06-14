#!/bin/bash
#SBATCH -A additya.popli
#SBATCH -n 5
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=5
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:1

module add cuda/8.0
module add cudnn/7-cuda-8.0

python pipeline.py --patch_types 1 --patch_size 96 --data_for experiment \
			--input_dir  $1 \
			--minu_dir  $2  \
			--output_dir $3 \
			--mode patches | tee $4.log
