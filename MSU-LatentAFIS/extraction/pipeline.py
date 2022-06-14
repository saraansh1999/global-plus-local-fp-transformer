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
import os 
import json 
import sys 
import torch 
import pickle 
import csv 
import math 
import utils 
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MAX = -np.inf
MIN = np.inf

def is_spoof_file(self, file):
								return ('Fake' in file or 'Spoof' in file or 'fake' in file or 'spoof' in file)


def read_predsmin(path, r, c):
								if not os.path.exists(path):
																return np.array([])

								mnt = np.array([])
								with open(path, 'rb') as f:
																file = pickle.load(f)
																mnt = file[b'mnt']
								mnt[:, -1] = 1
								return mnt

def read_vfmin(path, r, c):

																if not os.path.exists(path):
																																return np.array([])

																minu = []
																with open(path, 'r') as f:
																																reader = csv.reader(f, delimiter=' ')
																																for i, row in enumerate(reader):
																																																if i < 4:
																																																																continue
																																																w, h, ori, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3])/100
																																																if w >= c or h >= r:
																																																																continue
																																																minu.append([w, h, ori, conf])
																minu = np.asarray(minu, dtype=np.float32)
																I = np.argsort(minu[:, 3])
																I = I[::-1]
																minu = minu[I, :]
																#print(minu)
																return minu

def dtheta(a, b):
																if a-b <= math.pi and a-b >= -math.pi:
																																return np.abs(a-b)
																return 2*math.pi - np.abs(a-b)



class CLSExtractor:
																def __init__(self, des_model_dirs=None):
																								self.des_models = None
																								self.des_model_dirs = des_model_dirs
																								self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(ori_num=24)
																								self.des_models = []
																								for i, model_dir in enumerate(des_model_dirs):
																																																																																					print("Loading descriptor model (" + str(i+1) + " of " + str(len(des_model_dirs)) + "): " + model_dir)
																																																																																					self.des_models.append(descriptor.ImportGraph(model_dir, input_name="inputs:0",

																																																								 output_name='embedding:0'))
																								self.placeholder = tf.compat.v1.placeholder(tf.float32, (1, 384+2*args.p, 384+2*args.p, 1))
																								self.patch_extraction = tf.image.extract_image_patches(self.placeholder, [1, args.k, args.k, 1], [1, args.s, args.s, 1], [1, 1, 1, 1], padding='VALID')
																								print("init done")

																def get_results(self, img_file, image_dir, save_dir):
																								if not os.path.exists(img_file):
																																return None

																								start = timeit.default_timer()
																								img = io.imread(img_file, as_grey=True)
																								if len(img.shape) > 2:
																																img = rgb2gray(img)
																								if np.max(img) <= 1:
																																img *= 255
																								h, w = img.shape

																								# dp pre
																								img = preprocessing.pad_and_resize(img, 384, 384)

																								# generate patches
																								img = np.expand_dims(np.expand_dims(np.pad(img, ((args.p, args.p), (args.p, args.p)), 'constant'), axis=0), axis=-1)
																								patch_start = timeit.default_timer()
																								with tf.compat.v1.Session() as sess:
																																patches = sess.run(self.patch_extraction, feed_dict={self.placeholder:img})
																								patch_time = timeit.default_timer() - patch_start
																								patches = np.squeeze(patches)
																								patches = np.expand_dims(patches.reshape(patches.shape[0]*patches.shape[1], args.k, args.k), axis=-1)

																								# generate des
																								des = descriptor.cls_descriptor_extraction(patches, self.des_models, batch_size=128)

																								# save des
																								to_save = {
#                               'patches': patches,
																																'des': des
																								}
																								fname = os.path.join(save_dir, img_file[len(image_dir):])
																								fname = '.'.join(fname.split('.')[:-1]) + '.npy'
																								pardir = '/'.join(fname.split('/')[:-1])
																								if not os.path.exists(pardir):
																																os.makedirs(pardir)
																								np.save(fname, to_save, allow_pickle=True)
																								print(fname, img_file)
																								print(patch_time, timeit.default_timer() - start)
																								sys.stdout.flush()

																def run(self, image_dir, save_dir):
																								img_files = []
																								print(image_dir)
																								sys.stdout.flush()
																								for root, dirs, files in os.walk(image_dir):
																																for file in files:
																																								fpath = os.path.join(root, file)
																																								if file.endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')) and not is_spoof_file(fpath):
																																																img_files.append(fpath)
																																																#print(os.path.join(root, file))
																																																sys.stdout.flush()
																								print(len(img_files))
																								for img in img_files:
																																								self.get_results(img, image_dir, save_dir)

class GTExtractor:
																def __init__(self, patch_types=None, minu_model_dir=None, des_model_dirs=None):
																																self.des_models = None
																																self.patch_types = patch_types
																																self.minu_model = None
																																self.minu_model_dir = minu_model_dir
																																self.des_model_dirs = des_model_dirs

																																print("Loading models, this may take some time...")
																																if self.minu_model_dir is not None:
																																																print("Loading minutiae model: " + minu_model_dir)
																																																self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))
																																self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(ori_num=24)
																																patchSize = 160  #int(math.ceil(args.patch_size * (math.sqrt(2)))) #added formula for square outside circle outside patch  # 160 # 120

																																																																# print(">>>>>>>>>> CHANGE BACK TO 160")
																																oriNum = 64
																																if des_model_dirs is not None and len(des_model_dirs) > 0:
																																																self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

																																self.des_models = []

																																for i, model_dir in enumerate(des_model_dirs):

																																		print("Loading descriptor model (" + str(i+1) + " of " + str(len(des_model_dirs)) + "): " + model_dir)
																																		self.des_models.append(descriptor.ImportGraph(model_dir, input_name="inputs:0",
																																		 output_name='embedding:0'))

																																self.patch_size = args.patch_size
																																print("init done")

																def get_gt(self, img_file, image_dir, save_dir, ppi=500):
																																start_time = timeit.default_timer()
																																block_size = 16

																																if not os.path.exists(img_file):
																																																return None
																																img = io.imread(img_file, as_grey=True)
																																if ppi != 500:
																																																img = cv2.resize(img, (0, 0), fx=500.0/ppi, fy=500.0/ppi)

																																img = preprocessing.adjust_image_size(img, block_size)
																																if len(img.shape) > 2:
																																																img = rgb2gray(img)
																																if np.max(img) <= 1:
																																																img *= 255
																																h, w = img.shape

																																mask = get_maps.get_quality_map_intensity(img)
																																mnt = []
																																contrast_img = preprocessing.local_constrast_enhancement(img)
																																if args.minu_from == "afis":
																																																print("from afis")
																																																texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
																																																mnt = self.minu_model.run_whole_image(texture_img, minu_thr=args.thresh)
																																elif args.minu_from == "vf":
																																																print("from vf")
																																																min_name = os.path.join(args.minu_dir, img_file[len(image_dir):])
																																																min_name = '.'.join(min_name.split('.')[:-1]) + '.txt'
																																																print("min_name: ", min_name)
																																																mnt = read_vfmin(min_name, h, w)
																																																if len(mnt) == 0:
																																																																print("vf failed, skipping!")
																																																																return

																																elif args.minu_from == "preds":
																																																print("from preds")
																																																min_name = os.path.join(args.minu_dir, img_file[len(image_dir):])
																																																min_name = '.'.join(min_name.split('.')[:-1]) + '.pkl'
																																																print(min_name)
																																																mnt = read_predsmin(min_name, h, w)
																																																if len(mnt) == 0 and args.data_format == "minutiae_files":
																																																																print("skipping!")
																																																																return
																																if args.mode == "gt":
																																																start_time2 = timeit.default_timer()
																																																patches, des, times_patches, times_des = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models, self.patchIndexV,

																																batch_size=128, patch_size=self.patch_size, mode=args.mode)
#                                                                                               print("Time taken patches:", times_patches.item())
#                                                                                               print("Time taken des:", times_des.item())
#                                                                                               print("Time taken patch and des:", timeit.default_timer() - start_time2)
																																elif args.mode == "patches":
																																																patches = descriptor.minutiae_descriptor_extraction(img, mnt, self.patch_types, self.des_models, self.patchIndexV,

																																batch_size=128, patch_size=self.patch_size, mode=args.mode)
																																else:
																																																raise
																																print(img_file, img.shape, len(mnt), patches.shape)
																																dir_map, _ = get_maps.get_maps_STFT(img, patch_size=64, block_size=block_size, preprocess=True)

																																blkH = h // block_size
																																blkW = w // block_size
																																if args.data_format == 'minutiae_files':
																																																try:
																																																																NUM_MIN.append(patches.shape[1])
																																																																for i in range(patches.shape[1]):
																																																																																fname = os.path.join(save_dir, img_file[len(image_dir):])
																																																																																fname = '.'.join(fname.split('.')[:-1]) + '/' + str(i) + '.npy'
																																																																																pardir = '/'.join(fname.split('/')[:-1])
																																																																																if not os.path.exists(pardir):
																																																																																	os.makedirs(pardir)
																																																																																if args.mode == "gt":

																																																																																	to_save = {

																																																																																																	b'patch': patches[0, i],

																																																																																																	b'des': des[0, i],

																																																																																																	b'mnt': mnt[i]

																																																																																	}
																																																																																elif args.mode == "patches":

																																																																																	to_save = {

																																																																																																	b'patch': patches[0, i],

																																																																																																	b'mnt': mnt[i]

																																																																																	}
																																																																																else:
																																																																																	raise
																																																																																with open(fname, 'wb+') as fp:
																																																																																	pickle.dump(to_save, fp, protocol=2)
																																																																print(img_file)
																																																																to_save = {
																																																																																b'h': h,
																																																																																b'w': w,
																																																																																b'blkH': blkH,
																																																																																b'blkW': blkW,
																																																																																b'dir_map': dir_map,
																																																																																b'mask': mask
																																																																}
																																																																fname = os.path.join(save_dir, img_file[len(image_dir):])
																																																																fname = '.'.join(fname.split('.')[:-1]) + '/' + 'meta'
																																																																with open(fname, 'wb+') as fp:
																																																																																pickle.dump(to_save, fp, protocol=2)
																																																except Exception as e:
																																																																print("ERROR:", e)
																																else:
																																																if mnt.shape[0] == 0:
																																																	print("SKIPPED: NO MINUTIAE", img_file)
																																																	return;
																																																if args.mode == "gt":
																																																																to_save = {
																																																																	#															b'patches': patches,
																																																																																b'des': des,
																																																																																b'h': h,
																																																																																b'w': w,
																																																																																b'blkH': blkH,
																																																																																b'blkW': blkW,
																																																																																b'mnt': mnt,
																																																																																b'dir_map': dir_map,
																																																																																b'mask': mask
																																																																}
																																																elif args.mode == "patches":
																																																																to_save = {

																b'patches': patches,

																b'h': h,

																b'w': w,

																b'blkH': blkH,

																b'blkW': blkW,

																b'mnt': mnt,

																b'dir_map': dir_map,

																b'mask': mask
																																																																}
																																																else:
																																																																raise
																																																fname = os.path.join(save_dir, img_file[len(image_dir):])
																																																fname = '.'.join(fname.split('.')[:-1]) + '.npy'
																																																pardir = '/'.join(fname.split('/')[:-1])
																																																if not os.path.exists(pardir):

																																																	os.makedirs(pardir)
																																																np.save(fname, to_save, allow_pickle=True)
																																																print(fname, img_file)
																																																print(timeit.default_timer() - start_time)

																def run(self, image_dir, save_dir):
																																img_files = []
																																print(image_dir)
																																sys.stdout.flush()
																																for root, dirs, files in os.walk(image_dir):
																																																for file in files:
																																																																fpath = os.path.join(root, file)
																																																																if file.endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')):
																																																																																img_files.append(fpath)
																																																																																print(os.path.join(root, file))
																																																																																sys.stdout.flush()
																																print("#: ", len(img_files))
																																for img in img_files:
																																												try:
																																																self.get_gt(img, image_dir, save_dir)
																																												except:
																																																print("ERROR:", img)

class MinMapExtractor:
																def __init__(self, outsize = 138, patch_types=None, minu_model_dir=None):
																																self.patch_types = patch_types
																																self.minu_model = None
																																self.minu_model_dir = minu_model_dir
																																print("Loading models, this may take some time...")
																																if self.minu_model_dir is not None:
																																																print("Loading minutiae model: " + minu_model_dir)
																																																self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))
																																self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(ori_num=24)
																																self.patch_size = 96
																																patchSize = 160
																																oriNum = 64
																																self.outsize = outsize
																																#if des_model_dirs is not None and len(des_model_dirs) > 0:
																																#       self.patchIndexV = descriptor.get_patch_index(patchSize, patchSize, oriNum, isMinu=1)

																def get_patches(self, img_file, image_dir, save_dir, ppi=500):
																																block_size = 16
																																if not os.path.exists(img_file):
																																																return None
																																img = io.imread(img_file, s_grey=True)
																																if ppi != 500:
																																																img = cv2.resize(img, (0, 0), fx=500.0/ppi, fy=500.0/ppi)
																																# img = cv2.resize(img, (self.outsize, self.outsize), interpolation=cv2.INTER_CUBIC)

																																img = preprocessing.adjust_image_size(img, block_size)
																																#print(img.shape)
																																#print(img)
																																if len(img.shape) > 2:
																																																img = rgb2gray(img)
																																																img = img * 255
																																h, w = img.shape
																																#print(img)

																																start = timeit.default_timer()
																																mask = get_maps.get_quality_map_intensity(img)
																																# stop = timeit.default_timer()
																																# print('time for cropping : %f' % (stop - start))
																																# contrast_img = preprocessing.local_constrast_enhancement(img)
																																texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
																																mnt = np.array(self.minu_model.run_whole_image(texture_img, minu_thr=args.thresh))

																																sigmas = 1
																																# dx = 15 * img.shape[0] // self.outsize
																																# dy = 15 * img.shape[1] // self.outsize
																																# dx = dy = 7
																																minmap = np.zeros((self.outsize, self.outsize, 6))
																																# minmap = np.zeros((self.outsize, self.outsize, 6))
																																for i in range(mnt.shape[0]):
																																																x = int(mnt[i, 0] * self.outsize / img.shape[0])
																																																y = int(mnt[i, 1] * self.outsize / img.shape[1])
																																																ori = mnt[i, 2] % (math.pi * 2)

																																																for k in range(minmap.shape[2]):
																																																																co = np.exp(-dtheta(ori, (2*k*math.pi/6)) / (2 * (sigmas**2)))
																																																																# for i in range(max(0, x - 3), min(x + 3, minmap.shape[0])):
																																																																#       for j in range(max(0, y - 3), min(y + 3, minmap.shape[1])):
																																																																#               cs = np.exp(-(np.linalg.norm(np.array([j, i]) - np.array([y, x]))**2) / (2 * (sigmas**2)))
																																																																#               minmap[i, j, k] += cs*co
																																																																minmap[:, :, k] += co * utils.gaussian([self.outsize, self.outsize], [y, x])

																																for k in range(minmap.shape[2]):
																																																minmap[:, :, k] /= np.max(minmap[:, :, k]) + 1e-12
																																																minmap[:, :, k] *= 255
																																																# for i in range(max(0, x - dx), min(x + dx, minmap.shape[0])):
																																																#       for j in range(max(0, y - dy), min(y + dy, minmap.shape[1])):
																																																#               cs = np.exp(-(np.linalg.norm(np.array([j, i]) - np.array([y, x]))**2) / (2 * (sigmas**2)))
																																																#               minmap[i, j, k] += cs*co
																																																# if(np.max(minmap[:, :, k]) != 0):
																																																#       minmap[:, :, k] /= np.max(minmap[:, :, k])
																																																# minmap[:, :, k] *= 255

																																# newmmap = np.zeros((self.outsize, self.outsize, 6));
																																# newmmap[:, :, :3] = cv2.resize(minmap[:, :, :3], (self.outsize, self.outsize))
																																# newmmap[:, :, 3:] = cv2.resize(minmap[:, :, 3:], (self.outsize, self.outsize))

																																fname = os.path.join(save_dir, img_file[len(image_dir):])
																																fname = '.'.join(fname.split('.')[:-1]) + '.npy'
																																pardir = '/'.join(fname.split('/')[:-1])
																																if not os.path.exists(pardir):

																																	os.makedirs(pardir)
																																np.save(fname, minmap)
																																print(fname)


																def run(self, image_dir, save_dir):
																																img_files = []
																																for root, dirs, files in os.walk(image_dir):
																																																for file in files:
																																																																if file.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
																																																																																img_files.append(os.path.join(root, file))
																																for img in img_files:
																																																self.get_patches(img, image_dir, save_dir)


class MinExtractor:
																def __init__(self, patch_types=None, minu_model_dir=None):
																																self.patch_types = patch_types
																																self.minu_model = None
																																self.minu_model_dir = minu_model_dir
																																print("Loading models, this may take some time...")
																																if self.minu_model_dir is not None:
																																																print("Loading minutiae model: " + minu_model_dir)
																																																self.minu_model = (minutiae_AEC.ImportGraph(minu_model_dir))
																																self.dict, self.spacing, self.dict_all, self.dict_ori, self.dict_spacing = get_maps.construct_dictionary(ori_num=24)
																																self.patch_size = 96
																																patchSize = 160
																																oriNum = 64

																def get_patches(self, img_file, image_dir, save_dir, ppi=500):

																																block_size = 16
																																if not os.path.exists(img_file):
																																																return None
																																img = io.imread(img_file, as_gray=True)
																																if ppi != 500:
																																																img = cv2.resize(img, (0, 0), fx=500.0/ppi, fy=500.0/ppi)

																																img = preprocessing.adjust_image_size(img, block_size)
																																#print(img.shape)
																																#print(img)
																																if len(img.shape) > 2:
																																																img = rgb2gray(img)
																																																img = img * 255
																																h, w = img.shape
																																#print(img)

																																start = timeit.default_timer()
																																mask = get_maps.get_quality_map_intensity(img)
																																# stop = timeit.default_timer()
																																# print('time for cropping : %f' % (stop - start))
																																# contrast_img = preprocessing.local_constrast_enhancement(img)
																																texture_img = preprocessing.FastCartoonTexture(img, sigma=2.5, show=False)
																																mnt = self.minu_model.run_whole_image(texture_img, minu_thr=args.thresh)

																																fname = os.path.join(save_dir, img_file[len(image_dir):])
																																fname = '.'.join(fname.split('.')[:-1]) + '.txt'
																																pardir = '/'.join(fname.split('/')[:-1])
																																if not os.path.exists(pardir):
																																																os.makedirs(pardir)
																#       print('xmax: ', np.max(mnt[:, 1]))
																#       print('ymax: ', np.max(mnt[:, 0]))
																																with open(fname, 'w+') as f:
																																																writer = csv.writer(f, delimiter=' ')
																																																writer.writerow([int(w)])
																																																writer.writerow([int(h)])
																																																writer.writerow([int(args.res)]); # print("mnt: ", mnt)
																																																writer.writerow([int(mnt.shape[0])])
																																																for i in range(mnt.shape[0]):
																																																																# ??
																																																																#print(np.min(mnt[i, 2]), np.max(mnt[i, 2]))
																																																																global MAX
																																																																MAX = max(MAX, np.max(mnt[i, 2]))
																																																																global MIN
																																																																MIN = min(MIN, np.min(mnt[i, 2]))
																																																																ori = mnt[i, 2]
																																																																x = int(mnt[i, 0])
																																																																y = int(mnt[i, 1])
																																																																ori = ori % (math.pi * 2)
																																																																# ??
																																																																writer.writerow([x, y, ori])
																																# np.save(fname, mnt, allow_pickle=True)
																																print(fname, img_file, mnt.shape)

																def run(self, image_dir, save_dir):
																																img_files = []
																																for root, dirs, files in os.walk(image_dir):
																																																for file in files:
																																																																if file.endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')):
																																																																																img_files.append(os.path.join(root, file))
																																for img in img_files:
																																																self.get_patches(img, image_dir, save_dir)


class TemplateExtractor:
																def __init__(self, tft):
																																self.tft = tft
																																pass

																def get_templates(self, input_file, data_dir, template_dir):
																																print(input_file)
																																if self.tft == 'npy':
																																								dic = np.load(input_file, allow_pickle=True, encoding='bytes').item()
																																else:
																																								with open(input_file, 'rb') as fp:
																																																dic = pickle.load(fp)
																																print(dic[b'mnt'].shape)
																																start = timeit.default_timer()
																																dic[b'des'] = dic[b'des'][:1, :, :]
																																num_min = dic[b'des'].shape[1]
																																minu_template = template.MinuTemplate(h=dic[b'h'], w=dic[b'w'], \
																																																								blkH=dic[b'blkH'], blkW=dic[b'blkW'], \
																																																								minutiae=dic[b'mnt'], des=dic[b'des'], \
																																																								oimg=dic[b'dir_map'], mask=dic[b'mask'])

																																rolled_template = template.Template()
																																rolled_template.add_minu_template(minu_template)
																																fname = os.path.join(template_dir, input_file[len(data_dir):].lstrip('/'))
																																fname = '.'.join(fname.split('.')[:-1]) + '.dat'
																																pardir = '/'.join(fname.split('/')[:-1])
																																print(template_dir, fname, pardir)
																																if not os.path.exists(pardir):

																																	os.makedirs(pardir)
																																print(fname, input_file)
																																template.Template2Bin_Byte_TF_C(fname, rolled_template, isLatent=False)
																																tt = timeit.default_timer() - start
																																return num_min, tt

																def run(self, data_dir, save_dir):
																																data_files, times = [], []
																																for root, dirs, files in os.walk(data_dir):
																																																for file in files:
																																																																if file.endswith(('.'+self.tft)):
																																																																																data_files.append(os.path.join(root, file))
																																num_min, c = 0, 0
																																for file in data_files:
																																																print(file)
																																																ms, ts = self.get_templates(file, data_dir, save_dir)
																																																num_min += ms
																																																times.append(ts)
																																																c += 1
																																print("Avg \# min: ", num_min / c)
																																print("Avg time:", sum(times) / len(times))

'''
class DescriptorExtractor:

																def __init__(self, model_path, model_type, split_point):
																																self.model_type = model_type
																																if model_type == "student":
																																																																self.model = Inferencer(model_path, args.arch)
																																elif model_type == "joint":
																																																assert(split_point >= 1)
																																																self.model = Joint_Inferencer(split_point, model_path, args.arch)
																																elif model_type == "vf":
																																																self.model = vf_Inferencer(model_path, args.arch)
																																else:
																																																print("Invalid model type")

																																def get_descriptor(self, input_file, input_dir, save_dir):
																																																dic = np.load(input_file, allow_pickle=True, encoding='bytes').item()
																																																if self.model_type == "vf":
																																																																outputs = self.model.test(input_file, args.enhanced_dir)
																																																else:
																																																																inputs = dic[b'patches']
																																																																with torch.set_grad_enabled(False):
																																																																																outputs = self.model.test(inputs)
																																																fname = os.path.join(save_dir, input_file[len(input_dir):])
																																																fname = '.'.join(fname.split('.')[:-1]) + '.npy'
																																																pardir = '/'.join(fname.split('/')[:-1])
																																																if not os.path.exists(pardir):
																																																																os.makedirs(pardir)
																																																dic[b'des'] = outputs
																																																del dic[b'patches']
																																																print(fname, input_file)
																																																np.save(fname, dic, allow_pickle=True)
																																																assert(np.linalg.norm(dic[b'des'] - outputs) == 0)

																																def run(self, data_dir, save_dir):
																																																data_files = []
																																																for root, dirs, files in os.walk(data_dir):
																																																																for file in files:
																																																																																if file.endswith(('.npy')):

data_files.append(os.path.join(root, file))
																																																for file in data_files:
																																																																self.get_descriptor(file, data_dir, save_dir)
'''

def parse_arguments(argv):
																parser = argparse.ArgumentParser()

																parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
																parser.add_argument('--patch_types', type=str, default="1")
																parser.add_argument('--patch_size', type=int, default=96)
																parser.add_argument('--data_format', type=str, help='minutiae_file for one minutiae per npy file')
																parser.add_argument('--output_dir', type=str)
																parser.add_argument('--input_dir', type=str)
																parser.add_argument('--enhanced_dir', type=str, default=None)
																parser.add_argument('--minu_from', type=str, default="vf")
																parser.add_argument('--minu_dir', type=str, default=None)
																parser.add_argument('--thresh', type=float, default=0.15)
																parser.add_argument('--res', type=int, default=500)
																parser.add_argument('--mode', type=str)
																parser.add_argument('--k', type=int, default=96)
																parser.add_argument('--p', type=int, default=6)
																parser.add_argument('--s', type=int, default=13)
																parser.add_argument('--tft', type=str, default='pkl')
#       parser.add_argument('--model_path', type=str)
#       parser.add_argument('--model_type', type=str)
#       parser.add_argument('--arch', type=str)
#       parser.add_argument('--split_point', type=int, default=-1)
																return parser.parse_args(argv)


if __name__ == '__main__':
																args = parse_arguments(sys.argv[1:])
																print(args)
																if args.gpu:
																																os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

																if args.mode in ["patches", "gt", "min", "minmap", "cls"]:
																																import descriptor
																																import minutiae_AEC_modified as minutiae_AEC
																																import descriptor_PQ
																																import descriptor_DR
																																pwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
																																with open(pwd + '/afis.config') as config_file:
																																	config = json.load(config_file)
																																minu_model_dir = config['MinutiaeExtractionModel']

																if args.mode in ["patches", "gt"]:
																																types = args.patch_types.split(",")
																																patch_types = []
																																des_model_dirs = []
																																if "1" in types:
																																																model_dir = config['DescriptorModelPatch2']
																																																des_model_dirs.append(model_dir)
																																																patch_types.append(2)
																																if "2" in types:
																																																model_dir = config['DescriptorModelPatch8']
																																																des_model_dirs.append(model_dir)
																																																patch_types.append(8)
																																if "3" in types:
																																																model_dir = config['DescriptorModelPatch11']
																																																des_model_dirs.append(model_dir)
																																																patch_types.append(11)

																if args.mode == "patches":
																																TIMES = []
																																NUM_MIN = []
																																GTE = GTExtractor(patch_types=patch_types, des_model_dirs=des_model_dirs, minu_model_dir=minu_model_dir)
																																print("init run")
																																GTE.run(args.input_dir, args.output_dir)

																elif args.mode == "min":
																																ME = MinExtractor(minu_model_dir=minu_model_dir)
																																print(MIN, MAX)
																																ME.run(args.input_dir, args.output_dir)
																																print(MIN, MAX)

																elif args.mode == "minmap":
																																MME = MinMapExtractor(minu_model_dir=minu_model_dir)
																																MME.run(args.input_dir, args.output_dir)

																elif args.mode == "gt":
																																TIMES = []
																																NUM_MIN = []
																																GTE = GTExtractor(patch_types=patch_types, des_model_dirs=des_model_dirs,
																																																																 minu_model_dir=minu_model_dir)
																																GTE.run(args.input_dir, args.output_dir)
																																# print("Avg time: ", np.mean(np.array(TIMES)[25:75]))
																																# print("Avg no min: ", np.mean(np.array(NUM_MIN)[25:75]))

																elif args.mode == "templates":
																																import template
																																TE = TemplateExtractor(args.tft)
																																TE.run(args.input_dir, args.output_dir)


																elif args.mode == "cls":                # cvt local supervision
																								model_dir = config['DescriptorModelPatch2']
																								CLS = CLSExtractor(des_model_dirs=[model_dir])
																								CLS.run(args.input_dir, args.output_dir)
'''
																elif args.mode == "descriptors":
																																sys.path.append("/home/additya.popli/")
																																# from ijcb_patches import Inferencer, Joint_Inferencer
																																from SimCLR import vf_Inferencer
																																DE = DescriptorExtractor(args.model_path, args.model_type, args.split_point)
																																DE.run(args.input_dir, args.output_dir)
'''
