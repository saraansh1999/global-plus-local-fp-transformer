import os
import argparse
import numpy as np
from PIL import Image
import albumentations as A
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--sz', type=int)
args = parser.parse_args()

class DPPRE(object):

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.ts = A.Compose([
            A.PadIfNeeded(int(1.15 * s1), int(1.15 * s2), border_mode=cv2.BORDER_CONSTANT, value=255),
            A.CenterCrop(int(1.15 * s1), int(1.15 * s2)),
            A.Resize(int(1.07 * s1), int(1.07 * s2)),
            A.CenterCrop(s1, s2)])

dppre = DPPRE(args.sz, args.sz)

for root, dirs, files in os.walk(args.input_dir):
    for file in files:
        fpath = os.path.join(root, file)
        img = Image.open(fpath).convert('RGB')
        img = np.array(img)
        img = dppre.ts(image=img)['image']
        save_path = os.path.join(args.output_dir, fpath[len(args.input_dir.rstrip('/'))+1:]).split('.')[0] + '.png'
        pardir = '/'.join(save_path.split('/')[:-1])
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        cv2.imwrite(save_path, img)
        print(fpath, save_path)
