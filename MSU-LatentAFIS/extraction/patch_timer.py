from pipeline import read_vfmin
from skimage.color import rgb2gray
from skimage import io
import preprocessing
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import argparse
import shutil
import time
import descriptor

block_size = 16
img = io.imread('/home/saraansh.tandon/datasets/original_datasets/FVC_seg/cropped_images/FVC2004/Dbs/DB1_A/1/1_1.tif', as_grey=True)
img = preprocessing.adjust_image_size(img, block_size)
if len(img.shape) > 2:
	img = rgb2gray(img)
if np.max(img) <= 1:
	img *= 255
h, w = img.shape
minutiae = read_vfmin('/scratch/saraansh/datasets/FVC_seg_vfmin/cropped_images/FVC2004/Dbs/DB1_A/1/1_1.txt', h, w)

# warmup
for i in range(10):
	start = time.time()
	descriptor.extract_patches(minutiae, img, descriptor.get_patch_index(160, 160, 64, isMinu=1), 1)
	print("Time taken: ", time.time() - start)

times = []
for i in range(100):
	start = time.time()
	descriptor.extract_patches(minutiae, img, descriptor.get_patch_index(160, 160, 64, isMinu=1), 1)
	times.append(time.time() - start)

print("Avg time", sum(times) / len(times))
