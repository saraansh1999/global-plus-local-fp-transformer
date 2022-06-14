import argparse
import os
import numpy as np
import csv
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--mated_min_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

minutiae = {}
for root, dirs, files in os.walk(args.mated_min_dir):
	for file in files:
		fpath = os.path.join(root, file)

		i1 = file.split('.')[0].split('__')[0]
		i2 = file.split('.')[0].split('__')[1]
		print(fpath, i1, i2)

		if i1 not in minutiae:
			minutiae[i1] = set({})
		if i2 not in minutiae:
			minutiae[i2] = set({})

		with open(fpath) as f:
			reader = csv.reader(f, delimiter='\t')
			for i, row in enumerate(reader):
				if i < 4:
					continue
				m1 = row[:3]
				minutiae[i1].add(tuple(m1))
				m2 = row[3:]
				minutiae[i2].add(tuple(m2))

if os.path.exists(args.save_dir):
	shutil.rmtree(args.save_dir)	
os.makedirs(args.save_dir)

for k, s in minutiae.items():
	spath = os.path.join(args.save_dir, k+'.txt')
	s = np.array(list(s)).astype(np.float32)
	print(spath, s.shape)
	with open(spath, 'w+') as f:
		writer = csv.writer(f, delimiter=' ')
		writer.writerow([0])
		writer.writerow([0])
		writer.writerow([0])
		writer.writerow([0])
		for row in s:
			row[2] = 2*np.pi - (row[2]*np.pi/180)
			writer.writerow(row)

