#!/bin/bash
#SBATCH -A $USER
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=96:00:00
#SBATCH --mincpus=20
#SBATCH --nodelist=node01

module add cuda/8.0
module add cudnn/7-cuda-8.0

DATE=`date +"%d-%b-%Y"`

SAVE_NAME=cvt-13_detr_global_plus_local_${DATE}

#export CUDA_VISIBLE_DEVICES=0
bash run.sh -g 4 -t train --cfg experiments/global_plus_local.yaml \
OUTPUT_DIR /ssd_scratch/saraansh/git_models/${SAVE_NAME} \
DATASET.TRAIN_IMGS '/ssd_scratch/saraansh/data_seg/part1/cropped_images/Train/' \
DATASET.TRAIN_GLOBAL_EMBS '/ssd_scratch/saraansh/data_global_embs/part1/cropped_images/Train/' \
DATASET.TRAIN_TOKEN_EMBS '/ssd_scratch/saraansh/data_local_embs/part1/cropped_images/Train/' \
DATASET.VAL_IMGS '/ssd_scratch/saraansh/data_seg/part1/cropped_images/Val/' \
DATASET.VAL_GLOBAL_EMBS '/ssd_scratch/saraansh/data_global_embs/part1/cropped_images/Val/' \
DATASET.VAL_TOKEN_EMBS '/ssd_scratch/saraansh/data_local_embs/part1/cropped_images/Val/' \
 > logs/$SAVE_NAME 2>&1


