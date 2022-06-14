
from skimage.color import rgb2gray
from skimage import io
import numpy as np
import glob
import sys
import timeit
import argparse
import scipy
import cv2
import get_maps
import preprocessing
import descriptor
import os
import template
import minutiae_AEC_modified as minutiae_AEC
import json
import descriptor_PQ
import descriptor_DR
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FeatureExtraction_Rolled:
	def __init__(self, patch_types=None, des_model_dirs=None, minu_model_dir=None):
		self.des_models = None
		self.patch_types = patch_types
		self.minu_model = None
		self.minu_model_dir = minu_model_dir
		self.des_model_dirs = des_model_dirs

		print("Loading models, this may take some time...")
		if self.minu_model_dir is not None:
			print("Loading minutiae model: " + minu_model_dir)
			self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))

		self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(
			ori_num=24)
		patchSize = 160
		oriNum = 64
		if des_model_dirs is not None and len(des_model_dirs) > 0:
			self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

		if self.des_model_dirs is not None:
			self.des_models = []
			for i, model_dir in enumerate(des_model_dirs):
				print("Loading descriptor model (" + str(i+1) + " of " + str(len(des_model_dirs)) + "): " + model_dir)
				self.des_models.append(descriptor.ImportGraph(model_dir, input_name="inputs:0",
															  output_name='embedding:0'))
			self.patch_size = 96

	def remove_spurious_minutiae(self, mnt, mask):
		minu_num = len(mnt)
		if minu_num <= 0:
			return mnt
		flag = np.ones((minu_num,), np.uint8)
		h, w = mask.shape[:2]
		R = 5
		for i in range(minu_num):
			x = mnt[i, 0]
			y = mnt[i, 1]
			x = np.int(x)
			y = np.int(y)
			if x < R or y < R or x > w-R-1 or y > h-R-1:
				flag[i] = 0
			elif mask[y-R, x-R] == 0 or mask[y-R, x+R] == 0 or mask[y+R, x-R] == 0 or mask[y+R, x+R] == 0:
				flag[i] = 0
		mnt = mnt[flag > 0, :]
		return mnt

	def feature_extraction_single_rolled(self, image_dir, img_file, output_dir=None, ppi=500):
		#global IMG_LIV
		#global MIN_LIV
		#global IMG_FAKE
		#global MIN_FAKE

		global IMG
		global MIN

		block_size = 16

		if not os.path.exists(img_file):
			assert(0)
		# print(img_file)
		img = io.imread(img_file, s_grey=True)
		# print(img.shape)
		if ppi != 500:
			img = cv2.resize(img, (0, 0), fx=500.0/ppi, fy=500.0/ppi)

		img = preprocessing.adjust_image_size(img, block_size)
		if len(img.shape) > 2:
			img = rgb2gray(img) * 255
		h, w = img.shape
		# start = timeit.default_timer()
		mask = get_maps.get_quality_map_intensity(img)
		# stop = timeit.default_timer()
		# print('time for cropping : %f' % (stop - start))
		# start = timeit.default_timer()
		contrast_img = preprocessing.local_constrast_enhancement(img)
		texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
		mnt = self.minu_model.run_whole_image(texture_img, minu_thr=0.3)

		mnt = np.array(mnt)
		# print(img.shape, texture_img.shape, contrast_img.shape, mnt.shape)
		# contrast_path = os.path.join(output_dir, 'contrast_images', img_file[len(image_dir):].lstrip('/'))
		# texture_path  = os.path.join(output_dir, 'texture_images', img_file[len(image_dir):].lstrip('/'))
		minutiae_path = os.path.join(output_dir, 'minutiae', img_file[len(image_dir):].lstrip('/'))
		# patch_path = os.path.join(output_dir, 'patches', img_file[len(image_dir):].lstrip('/'))
		# descriptor_path = os.path.join(output_dir, 'descriptor', img_file[len(image_dir):].lstrip('/'))

		minutiae_path = '.'.join(minutiae_path.split('.')[:-1] + ['npy'])
		# patch_path = '.'.join(patch_path.split('.')[:-1] + ['npy'])
		# descriptor_path = '.'.join(descriptor_path.split('.')[:-1] + ['npy'])

		# contrast_dir = '/'.join(contrast_path.split('/')[:-1])
		# if not os.path.exists(contrast_dir):
		#	os.makedirs(contrast_dir)

		# texture_dir  = '/'.join(texture_path.split('/')[:-1])
		# if not os.path.exists(texture_dir):
		#	os.makedirs(texture_dir)

		minutiae_dir = '/'.join(minutiae_path.split('/')[:-1])
		if not os.path.exists(minutiae_dir):
			os.makedirs(minutiae_dir)

		# patch_dir = '/'.join(patch_path.split('/')[:-1])
		# if not os.path.exists(patch_dir):
		#	os.makedirs(patch_dir)

		# descriptor_dir = '/'.join(descriptor_path.split('/')[:-1])
		# if not os.path.exists(descriptor_dir):
		#	os.makedirs(descriptor_dir)

		# print(contrast_path, texture_path, minutiae_path)
		# print(contrast_img.shape, texture_img.shape, mnt.shape)
		# print(contrast_img.dtype, texture_img.dtype, mnt.dtype)

		# Image.fromarray(np.array(contrast_img, dtype=np.uint8)).save(contrast_path)
		# Image.fromarray(np.array(texture_img, dtype=np.uint8)).save(texture_path)
		# np.savetxt(minutiae_path, mnt)

		IMG = IMG + 1
		MIN = MIN + mnt.shape[0]

		# stop = timeit.default_timer()
		# print('time for minutiae : %f' % (stop - start))
		# print(img_file, len(mnt))
		# start = timeit.default_timer()
		# patches, des = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models, self.patchIndexV,
		#												batch_size=256, patch_size=self.patch_size, mode='gt')

		np.save(minutiae_path, mnt)
		# np.save(patch_path, patches)
		# np.save(descriptor_path, des)


		# stop = timeit.default_timer()
		# print('time for descriptor : %f' % (stop - start))
		# print(patches.shape, des.shape)
		#if(img_file.find("Live") != -1):
		#	IMG_LIV += 1
		#	MIN_LIV += patches.shape[1]
		#else:
		#	IMG_FAKE += 1
		#	MIN_FAKE += patches.shape[1]
		#dic = {0: '2', 1:'8', 2:'11'}
		'''
		for idx in range(patches.shape[0]):
			out_dir = os.path.join(output_dir, dic[idx], img_file[len(image_dir):].lstrip('/'))
			# print(out_dir)
			out_dir = '.'.join(out_dir.split('.')[:-1])
			os.makedirs(out_dir)
			# print(image_dir, img_file, out_dir)
			# print(img_file, "Total:", patches[idx].shape[0])
			sys.stdout.flush()
			for i in range(patches[idx].shape[0]):
				d = {
					'patch': patches[idx, i, ...],
					'des': des[idx, i, ...]
					}
				save_name = os.path.join(out_dir, str(i) + '.npy')
				np.save(save_name, d, allow_pickle=True)

				# print('patch:', d['patch'].shape)
				# print('des:', d['des'].shape)
		'''

	def feature_extraction(self, image_dir, output_dir=None, enhancement=False):


		#if not os.path.exists(os.path.join(output_dir, '2')):
		#	os.makedirs(os.path.join(output_dir, '2'))
		#if not os.path.exists(os.path.join(output_dir, '8')):
		#	os.makedirs(os.path.join(output_dir, '8'))
		#if not os.path.exists(os.path.join(output_dir, '11')):
		#	os.makedirs(os.path.join(output_dir, '11'))

		img_files = []
		for root, dirs, files in os.walk(image_dir):
			for file in files:
				if file.endswith(('.png', '.bmp', '.jpg', '.jpeg')):
					img_files.append(os.path.join(root, file))

		assert(len(img_files) > 0)
		print('total files:', len(img_files))

		for i, img_file in enumerate(img_files):
			#try:
			self.feature_extraction_single_rolled(image_dir, img_file, output_dir=output_dir)
			#except Exception as e:
			#	print('>>>>>> Error', img_file, e)
			if i % 500 == 0:
				print('done!', i)


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
	parser.add_argument('--N1', type=int, help='rolled index from which the enrollment starts', default=0)
	parser.add_argument('--N2', type=int, help='rolled index from which the enrollment starts', default=2000)
	parser.add_argument('--tdir', type=str, help='data path for minutiae descriptor and minutiae extraction')
	parser.add_argument('--idir', type=str, help='data path for images')
	return parser.parse_args(argv)


if __name__ == '__main__':

	IMG = 0
	MIN = 0

	args = parse_arguments(sys.argv[1:])
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	pwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	with open(pwd + '/afis.config') as config_file:
		config = json.load(config_file)

	des_model_dirs = []
	patch_types = []
	# model_dir = config['DescriptorModelPatch2']
	# des_model_dirs.append(model_dir)
	# patch_types.append(2)
	#model_dir = config['DescriptorModelPatch8']
	#des_model_dirs.append(model_dir)
	#patch_types.append(8)
	#model_dir = config['DescriptorModelPatch11']
	#des_model_dirs.append(model_dir)
	#patch_types.append(11)

	minu_model_dir = config['MinutiaeExtractionModel']

	LF_rolled = FeatureExtraction_Rolled(patch_types=patch_types, des_model_dirs=des_model_dirs,
										 minu_model_dir=minu_model_dir)

	image_dir = args.idir if args.idir else config['GalleryImageDirectory']
	template_dir = args.tdir if args.tdir else config['GalleryTemplateDirectory']
	print("Starting feature extraction (batch)...")
	LF_rolled.feature_extraction(image_dir=image_dir, output_dir=template_dir, enhancement=False)
	print("STATS: ", IMG, MIN)
	#print("Fake: ", IMG_FAKE, MIN_FAKE)
	# print("Finished feature extraction. Starting dimensionality reduction...")
	# descriptor_DR.template_compression(input_dir=template_dir, output_dir=template_dir,
									   # model_path=config['DimensionalityReductionModel'],
									   # isLatent=False, config=None)
	# print("Finished dimensionality reduction. Starting product quantization...")
	# descriptor_PQ.encode_PQ(input_dir=template_dir, output_dir=template_dir, fprint_type='rolled')
	# print("Finished product quantization. Exiting...")

