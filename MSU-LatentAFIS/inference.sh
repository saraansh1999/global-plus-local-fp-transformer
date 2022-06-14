#!/bin/bash

for i in 0
do
	echo "---------------For channel ${i} ----------------"

	echo "Deleting old files..."
	rm -rf /scratch/srnsh_try

	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate latentafis

	# get templates
	echo "Template Extraction..."
	python extraction/pipeline.py --gpu 0 --input_dir $1/ \
										--output_dir /scratch/srnsh_try/templates_channels/patchtype_$i/ \
										--mode templates --tft $5

	conda deactivate
        source ~/patch_CvT/cvt/bin/activate

	# get metrics
	echo "Metrics Calculation..."
	python metric_extractor.py --dir /scratch/srnsh_try/templates_channels/patchtype_$i/ \
										--people $3 --impressions $4 --savename $2

done

# $1 -> "path"
# $2 -> "log savename"
# $3 -> "people"
# $4 -> "impressions"
# $5 -> "type of input file for template (pkl/npy)"
