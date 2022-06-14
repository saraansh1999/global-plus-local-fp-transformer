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
parser.add_argument('--save_dir')
parser.add_argument('--w_des', type=float, default=1)
parser.add_argument('--w_mnt', type=float, default=1)
parser.add_argument('--use_ori', type=bool, default=True)
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

class HungarianMatcher():
	def __init__(self, w_kd: float = 1, w_posori: float = 1):
		super().__init__()
		self.w_kd = w_kd
		self.w_posori = w_posori
		assert w_kd != 0 or w_posori != 0, "all costs cant be 0"

	def __call__(self, outputs, targets):
		if args.use_ori:
			slice_ind = 3
		else:
			slice_ind = 2
		out_embs = torch.tensor(outputs[b'des'].squeeze()).double()
		out_posori = torch.tensor(outputs[b'mnt'][:, :slice_ind]).double()

		tgt_embs = torch.tensor(targets[b'des'][0].squeeze()).double()
		tgt_posori = torch.tensor(targets[b'mnt'][:, :slice_ind]).double()

		cost_kd = torch.cdist(out_embs, tgt_embs, p=2)

		cost_posori = torch.cdist(out_posori, tgt_posori, p=1)

		C = self.w_kd * cost_kd + self.w_posori * cost_posori
		r, c = linear_sum_assignment(C)
		return r, c

matcher = HungarianMatcher(args.w_des, args.w_mnt)
dppre = DPPRE(384, 384)
l = 15
done = 0
for root, dirs, files in os.walk(args.image_dir):
    for file in files:
        plt.figure()
        fpath = os.path.join(root, file)
        img = Image.open(fpath).convert('RGB')

        gt_path = os.path.join(args.gt_mnt_dir, fpath[len(args.image_dir.rstrip('/'))+1:]).split('.')[0] + '.npy'
        gt_dict = np.load(gt_path, allow_pickle=True, encoding='bytes').item()
        gt = gt_dict[b'mnt'][:, :3]

        transformed = dppre.ts(image=np.array(img), keypoints=gt)
        inimg = transformed['image']
        gt = np.array(transformed['keypoints'])
        gt_dict[b'mnt'] = gt

        plt.imshow(inimg)

        plt.scatter(gt[:, 0], gt[:, 1], color='limegreen', s=30, alpha=0.5)
        for i in range(gt.shape[0]):
            plt.plot((gt[i, 0], gt[i, 0] + l * np.cos(2*np.pi-gt[i, 2])), (gt[i, 1], gt[i, 1] + l * np.sin(2*np.pi-gt[i, 2])), color='limegreen', alpha=0.5)

        pred_path = os.path.join(args.pred_mnt_dir, fpath[len(args.image_dir.rstrip('/'))+1:]).split('.')[0] + '.pkl'
        pred_dict = pkl.load(open(pred_path, 'rb'))
        pred = pred_dict[b'mnt']
        print(pred.shape)
        plt.scatter(pred[:, 0], pred[:, 1], color='red', s=30)
        for i in range(pred.shape[0]):
            plt.plot((pred[i, 0], pred[i, 0] + l * np.cos(2*np.pi-pred[i, 2])), (pred[i, 1], pred[i, 1] + l * np.sin(2*np.pi-pred[i, 2])), color='red')

        r, c = matcher(pred_dict, gt_dict)
        for i in range(len(r)):
            plt.plot([pred_dict[b'mnt'][r[i], 0], gt_dict[b'mnt'][c[i], 0]], [pred_dict[b'mnt'][r[i], 1], gt_dict[b'mnt'][c[i], 1]], color='black')

        save_path = os.path.join(args.save_dir, fpath[len(args.image_dir.rstrip('/'))+1:]).split('.')[0] + '.png'
        pardir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        plt.savefig(save_path)
        done += 1
        print(done)
        if done  >= 50:
            exit(0)
