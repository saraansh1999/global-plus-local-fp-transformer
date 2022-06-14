import sys
import cv2
import os
import numpy as np
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
import matplotlib.pyplot as plt
from scipy.ndimage import morphology as morph
from datetime import datetime
import traceback
from PIL import Image, ImageEnhance
import argparse

def translate(img):
        w, h = img.size
        hor = np.random.randint(low=-w//3, high=w//3)
        ver = np.random.randint(low=-h//3, high=h//3)
        img = img.transform(img.size, Image.AFFINE, (1, 0, hor, 0, 1, ver), \
                                fillcolor='white')
        return img

def rotate(img):
        deg = np.random.uniform(low=-20, high=20)
        img = img.rotate(deg, fillcolor='white')
        return img

def quality(img):
        # quality
        contrast = ImageEnhance.Contrast(img)
        con_val = np.random.uniform(low=0.6, high=1.4)
        img = contrast.enhance(con_val)

        brightness = ImageEnhance.Brightness(img)
        bri_val = np.random.uniform(low=0.7, high=1.3)
        img = brightness.enhance(bri_val)

        sharpness = ImageEnhance.Sharpness(img)
        shp_val = np.random.uniform(low=0.4, high=1.6)
        img = sharpness.enhance(shp_val)

        return img

def crop(img):
        w, h = img.size
        crop_w, crop_h = int(0.8*w), int(0.8*h)
        x1 =  np.random.randint(0, w - crop_w)
        y1 = np.random.randint(0, h - crop_h)
        img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        return img

def check_path(fname):
    dir_name = '/'.join(fname.split('/')[:-1])
    if not os.path.exists(dir_name):
        print("Creating Directory: ", dir_name)
        os.makedirs(dir_name)


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', action='store', required=True, help='Input path')
parser.add_argument('--output_dir', action='store', required=True, help='Output path')
parser.add_argument('--morph_close_sz', default=20, type=int)
parser.add_argument('--segment', action='store_true', help='to segment or not?')
parser.add_argument('--augment', action='store_true', help='to augment or not?')
parser.add_argument('--augmentations', help='List of potential augs', default="")
parser.add_argument('--replications', help='no of replications per image', default=5, type=int)
args = parser.parse_args()

dataset_dir = args.input_dir.rstrip('/')
base_str = args.output_dir.rstrip('/')
image_dir = os.path.join(base_str, 'cropped_images/')

augmentations = args.augmentations.split(',')
err_cnt = 0
cnt = 0
for root, dirs, files in os.walk(dataset_dir):
    if len(files) > 0:
        print("Starting: ", root)

        for file in files:

            src_d = os.path.join(root, file)
            if not os.path.isfile(src_d):
                continue

            try:
#            if 1:
                image_d = os.path.join(image_dir, src_d[len(dataset_dir) + 1:])
                check_path(image_d)

                print("----------------\n", src_d)
                sys.stdout.flush()

                im = cv2.imread(src_d)
                
                if args.segment:
                    # Segmentation
                    ###########################################################################
                    print("Segmenting...")
                    original_im = im.copy()
                    if len(im.shape) > 2:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('temp.png', original_im)
                    im = 255 - im
                    threshold_global_otsu = threshold_otsu(im)
                    im = np.asarray((im > threshold_global_otsu), dtype='uint8')

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.morph_close_sz, args.morph_close_sz))
                    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                    im = cv2.dilate(im, kernel)
                    im = im * 255

                    final_mask = Image.fromarray(im)
                    component_count, component_mask = cv2.connectedComponents(im)
                    label, label_frq = np.unique(component_mask, return_counts=True)
                    needed_labels = label[label_frq.argsort()[-2:][::-1]]
                    X_sum, Y_sum, final_label = np.sum(im[np.where(component_mask == needed_labels[0])]), np.sum(im[np.where(component_mask == needed_labels[1])]), 0
                    if X_sum > Y_sum:
                        final_label = needed_labels[0]
                    else:
                        final_label = needed_labels[1]
                    X_idx, Y_idx = np.where(component_mask == final_label)[0], np.where(component_mask == final_label)[1]
                    (x_min, y_min, x_max, y_max) = (np.min(X_idx), np.min(Y_idx), np.max(X_idx), np.max(Y_idx))
                    im = im[x_min:x_max, y_min:y_max]
                    original_im = original_im[max(x_min-7, 0):min(x_max+7, original_im.shape[0]), max(y_min-7, 0):min(y_max+7, original_im.shape[1])]
                    cv2.imwrite('temp1.png', original_im)

                    final_im = Image.fromarray(original_im)

                    cropped_mask = Image.fromarray(im)

                    ###########################################################################
                else:
                    final_im = Image.fromarray(im)

                # Augmentation
                ###########################################################################
                if args.augment:
                    print("Augmenting...", augmentations)
                    augs = np.random.choice(augmentations, args.replications, replace=True).tolist()
                    for i in range(len(augs)):
                        save_path = '.'.join(image_d.split('.')[:-1]) + '/' + str(i) + '.' + image_d.split('.')[-1]

                        if augs[i] == "quality":
                                res = quality(final_im)
                        elif augs[i] == "translate":
                                res = translate(final_im)
                        elif augs[i] == "rotate":
                                res = rotate(final_im)
                        elif augs[i] == "crop":
                                res = crop(final_im)

                        print(save_path)
                        pardir = '/'.join(save_path.split('/')[:-1])
                        if not os.path.exists(pardir):
                            os.makedirs(pardir)
                        res.save(save_path)
                else:
                    print(image_d)
                    Image.fromarray(original_im).save(image_d)
                ###########################################################################



            except Exception as e:
#            else:
                err_cnt += 1
                print("Error: ", src_d, traceback.print_exc(), e)

            cnt += 1
            print(cnt, "done")

print("Number of errors: ", err_cnt)
