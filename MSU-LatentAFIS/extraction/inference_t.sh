#!/bin/bash


# get metrics
echo "Metrics Calculation..."
python metric_extractor.py --dir /scratch/additya/templates_all/ \
									--people 5 --impressions 12 --savename db2a_all | tee results_db2a_all.txt

for i in 0 1 2
do
	echo "---------------For channel ${i} ----------------"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate latentafis
	# get templates
	echo "Template Extraction..."
	python extraction/pipeline.py --gpu 0 --input_dir /scratch/additya/gt_channels/patchtype_$i/ \
										--output_dir /scratch/additya/templates_channels/patchtype_$i/ \
										--mode templates

	conda activate base

	# get metrics
	echo "Metrics Calculation..."
	python metric_extractor.py --dir /scratch/additya/templates_channels/patchtype_$i/ \
										--people 140 --impressions 12 --savename db2a_$i | tee results_db2a_$i.txt

done





# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate latentafis

# # get patches
# if [[ "$1" == "NA" ]];
# then
# 	echo "No Patch Extraction"
# else
# 	echo "Patch Extraction..."
# 	python extraction/pipeline.py --gpu $8 --input_dir $1 --output_dir $2 \
# 									--mode patches --patch_types 1,2,3
# fi

# # get descriptors
# echo "Descriptor Extraction..."
# python extraction/pipeline.py --gpu $8 --input_dir $2 --output_dir $3 \
# 									--mode descriptors --model_path $9

# # get templates
# echo "Template Extraction..."
# python extraction/pipeline.py --gpu $8 --input_dir $3 --output_dir $4 \
# 									--mode templates

# conda activate base

# # get metrics
# echo "Metrics Calculation..."
# python metric_extractor.py --dir $4 --people $5 --impressions $6 --savename $7



#Command with patch extraction
# bash inference.sh /scratch/additya/temp/ /scratch/additya/patches/ /scratch/additya/descriptors/ /scratch/additya/templates/ 140 12 infer 0 /home/additya.popli/12000.pth 

#Command without patch extraction
# bash inference.sh NA /scratch/additya/patches/ /scratch/additya/descriptors/ /scratch/additya/templates/ 140 12 infer 0

