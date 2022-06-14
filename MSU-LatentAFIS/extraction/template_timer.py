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
import sys
import torch
import pickle
import csv
import math
import utils, time
sys.path.append("/home/additya.popli/")
from ijcb_patches import Inferencer, Joint_Inferencer

times = []
class TemplateExtractor:
                def __init__(self):
                                pass

                def get_templates(self, input_file, data_dir, template_dir):
                                print(input_file)
                                dic = np.load(input_file, allow_pickle=True, encoding='bytes').item()
                                #fp = open(input_file, 'rb')
                                #dic = pickle.load(fp)
                                start = time.time()
                                dic[b'des'] = dic[b'des'][:1, :, :]
                                minu_template = template.MinuTemplate(h=dic[b'h'], w=dic[b'w'], blkH=dic[b'blkH'], blkW=dic[b'blkW'], minutiae=dic[b'mnt'], des=dic[b'des'], oimg=dic[b'dir_map'],

                                                                          mask=dic[b'mask'])

                                rolled_template = template.Template()
                                rolled_template.add_minu_template(minu_template)
                                times.append(time.time() - start)
                                fname = os.path.join(template_dir, input_file[len(data_dir):].lstrip('/'))
                                fname = '.'.join(fname.split('.')[:-1]) + '.dat'
                                pardir = '/'.join(fname.split('/')[:-1])
                                # print(template_dir, fname, pardir)
                                if not os.path.exists(pardir):
                                                os.makedirs(pardir)
                                print(fname, input_file)
                                start = time.time()
                                template.Template2Bin_Byte_TF_C(fname, rolled_template, isLatent=False)
                                times[-1] += time.time() - start
#				print(times[-1])

                def run(self, data_dir, save_dir):
                                data_files = []
                                for root, dirs, files in os.walk(data_dir):
                                                for file in files:
                                                                if file.endswith(('.npy')):
                                                                                data_files.append(os.path.join(root, file))
                                for file in data_files:
                                                self.get_templates(file, data_dir, save_dir)


def parse_arguments(argv):
                parser = argparse.ArgumentParser()
                parser.add_argument('--output_dir', type=str)
                parser.add_argument('--input_dir', type=str)
                return parser.parse_args(argv)


if __name__ == '__main__':
        args = parse_arguments(sys.argv[1:])
        TE = TemplateExtractor()
        TE.run(args.input_dir, args.output_dir)
	times = np.array(times)
        #print(times.shape)
        print("Time take: ", np.mean(times) * 1000, "ms")
