import copy
import cv2
import albumentations as A
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl
import torch
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir')
parser.add_argument('--gt_mnt_dir')
parser.add_argument('--pred_mnt_dir')
parser.add_argument('--pred_type')
parser.add_argument('--alpha', type=float, default=20)
args = parser.parse_args()

class DPPRE(object):

	def __init__(self, s1, s2):
		self.s1 = s1
		self.s2 = s2
		self.ts = A.Compose([
			A.PadIfNeeded(int(1.15 * s1), int(1.15 * s2), border_mode=cv2.BORDER_CONSTANT, value=255),
			A.CenterCrop(int(1.15 * s1), int(1.15 * s2)),
			A.Resize(int(1.07 * s1), int(1.07 * s2)),
			A.CenterCrop(s1, s2)],
			keypoint_params=A.KeypointParams('xya', remove_invisible=False, angle_in_degrees=False))

dppre = DPPRE(384, 384)
l = 15
print(args.image_dir, args.pred_type)
gt_tot, tot, imgs, avg_s, avg_m, avg_p, avg_ep, avg_gi = 0, 0, 0, 0, 0, 0, 0, 0
for root, dirs, files in os.walk(args.image_dir):
	for file in files:
		fpath = os.path.join(root, file)
		img = Image.open(fpath).convert('RGB')

		gt_path = os.path.join(args.gt_mnt_dir, fpath[len(args.image_dir.rstrip('/'))+1:]).split('.')[0] + '.npy'
		if not os.path.exists(gt_path):
			continue
		gt_dict = np.load(gt_path, allow_pickle=True, encoding='bytes').item()
		gt = gt_dict[b'mnt'][:, :3]
		transformed = dppre.ts(image=np.array(img), keypoints=gt)
		inimg = transformed['image']
		vf = np.array(transformed['keypoints'])

		if args.pred_type == 'our':
			pred_path = os.path.join(args.pred_mnt_dir, fpath[len(args.image_dir.rstrip('/'))+1:]).split('.')[0] + '.pkl'
			if not os.path.exists(pred_path):
				continue
			pred_dict = pkl.load(open(pred_path, 'rb'))
			pred = pred_dict[b'mnt'][:, :3].tolist()
		elif args.pred_type == 'mindtct':
			inds = np.argwhere(vf[:, 2] >= np.pi)
			vf[inds, 2] -= np.pi
			pred_path = os.path.join(args.pred_mnt_dir, fpath[len(args.image_dir.rstrip('/'))+1:]).split('.')[0] + '.xyt'
			f = open(pred_path, 'r')
			lines = f.readlines()
			pred = []
			for i in range(len(lines)):
				m = lines[i].split()
				m = [int(x) for x in m]
				m[2] = m[2] * np.pi / 180
				pred.append(m[:3])
			transformed = dppre.ts(image=np.array(img), keypoints=pred)
			pred = transformed['keypoints']


		imgs += 1
		pairs, spurious, missing = [], copy.deepcopy(pred), []
		gt_tot += vf.shape[0]
		tot += len(spurious)
		for fg in vf:
			best_d = np.inf
			best = None
			for fd in pred:
				d = np.linalg.norm(fg[:2] - fd[:2])
				if d < args.alpha:
					if d < best_d:
						best_d = d; best = fd
			if best_d < np.inf and best in spurious:
				spurious.remove(best)
				pairs.append([fg, np.array(best)])
			else:
				missing.append(fg)

		pairs = np.array(pairs)
		avg_p += pairs.shape[0]
		avg_s += len(spurious)
		avg_m += len(missing)
		ep = np.sqrt(np.sum((pairs[:, 0] - pairs[:, 1])[:, :2]**2) / pairs.shape[0])
		avg_ep += ep
		ori_diff = (pairs[:, 0] - pairs[:, 1])[:, 2]
		inds = np.argwhere(ori_diff < -np.pi)
		ori_diff[inds] += 2*np.pi
		inds = np.argwhere(ori_diff >= np.pi)
		ori_diff[inds] -= 2*np.pi
		gi = (pairs.shape[0] - min(2*vf.shape[0], len(spurious)) - len(missing)) / vf.shape[0]
		avg_gi += gi

		if imgs % 100 == 0:
			print(imgs, "done.")

print("Avg \# gt mnt", gt_tot / imgs)
print("Avg \# pred mnt", tot / imgs)
print("Avg \# pairs", avg_p / imgs)
print("Avg \# spurious", avg_s / imgs)
print("Avg \# missing", avg_m / imgs)
print("Avg ep:", avg_ep / imgs)
print("Avg gi:", avg_gi / imgs)
