#!/bin/bash

if [[ "$1" == "1" ]];
then
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate latentafis

	# get patches
	echo "---------------- GT extraction---------------------"
	python extraction/pipeline.py --gpu 0 --input_dir /scratch/additya/fvc2000/DB1_A_segmented/ \
										--output_dir /scratch/additya/fvc2000/db1a_gt_all/ \
										--mode gt --patch_types 1,2,3

	# split by patchtypes
	echo "--------------------Channel Splitting-------------------------"
	python split_channels.py --input_dir /scratch/additya/fvc2000/db1a_gt_all/ \
										--output_dir /scratch/additya/fvc2000/db1a_gt_channels/
else
	echo "No patch extraction"
fi



# echo "---------------For all channels----------------"
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate latentafis

# #get descriptors
# echo "Descriptor Extraction..."
# python extraction/pipeline.py --gpu 0 --input_dir /scratch/additya/gt_all/ \
# 									--output_dir /scratch/additya/des_all_$2/ \
# 									--mode descriptors \
# 									--model_path $3 \
# 									--model_type $2

# # get templates
# echo "Template Extraction..."
# python extraction/pipeline.py --gpu 0 --input_dir /scratch/additya/des_all_$2/ \
# 									--output_dir /scratch/additya/templates_all_$2/ \
# 									--mode templates

# conda activate base

# # get metrics
# echo "Metrics Calculation..."
# python metric_extractor.py --dir /scratch/additya/templates_all_$2/ \
# 									--people 140 --impressions 12 --savename model_inf_db2a_all_$2_$4_$5 | tee model_inf_results_db2a_all_$2_$4_$5_$6.txt
# mv model_inf_results_db2a_all_$2_$3_$4_$5_$6.txt model_infs/

# for i in 0 1 2
for i in 0
do
	echo "---------------For channel ${i} ----------------"

	echo "Deleting old files..."
	rm -rf /scratch/additya_try

	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate latentafis

	#get descriptors
	echo "Descriptor Extraction..."
	python extraction/pipeline.py --gpu 0 --input_dir /scratch/additya/gt_channels/patchtype_$i/ \
										--output_dir /scratch/additya_try/des_channels_$2/patchtype_$i/ \
										--mode descriptors \
										--model_path $3 \
										--model_type $2 \
										--arch $9 \
										--split_point $4 | tee -a fvc00db1a_$9_${10}_$2_sensor$5_fpad$7_vf$8_sp$4_run$6.txt

	# get templates
	echo "Template Extraction..."
	python extraction/pipeline.py --gpu 0 --input_dir /scratch/additya_try/des_channels_$2/patchtype_$i/ \
										--output_dir /scratch/additya_try/templates_channels_$2/patchtype_$i/ \
										--mode templates | tee -a fvc00db1a_$9_${10}_$2_sensor$5_fpad$7_vf$8_sp$4_run$6.txt

	conda activate base

	# get metrics
	echo "Metrics Calculation..."
	python metric_extractor.py --dir /scratch/additya_try/templates_channels_$2/patchtype_$i/ \
										--people 140 --impressions 12 --savename fvc00db1a_$9_${10}_$2_sensor$5_fpad$7_vf$8_sp$4_run$6 | tee -a fvc00db1a_$9_${10}_$2_sensor$5_fpad$7_vf$8_sp$4_run$6.txt
	mv fvc00db1a_$9_${10}_$2_sensor$5_fpad$7_vf$8_sp$4_run$6.txt sept_exps/

done

# $1 -> "0 / 1 == no patches / patches"
# $2 -> "student / joint"
# $3 -> "model path"
# $4 -> "split point (-1 / >=1)"
# $5 -> "sensor"
# $6 -> "run no."
# $7 -> "fpad"
# $8 -> "vf"
# $9 -> "arch"
# ${10} -> "train data"
