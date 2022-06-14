import argparse
import os
import numpy as np
import csv
from pyeer.eer_info import get_eer_stats
from scipy.spatial import distance

parser = argparse.ArgumentParser()
parser.add_argument('--mated_min_dir', type=str)
parser.add_argument('--des_dir', type=str)
parser.add_argument('--save_prefix', type=str)
parser.add_argument('--impressions', type=int)
parser.add_argument('--subjects', type=int)
args = parser.parse_args()

def normalize_scores(true_scores, false_scores):
	mini = min(np.min(true_scores), np.min(false_scores))
	maxi = max(np.max(true_scores), np.max(false_scores))
	tsn = (true_scores - mini) / (maxi - mini)
	fsn = (false_scores - mini) / (maxi - mini)
	return tsn, fsn

def get_similarity(x, y):
	return 1 - distance.cosine(x, y)

def calculate_metrics(ts, fs, title):
	print(">" * 20, title)
	res = get_eer_stats(ts, fs)
	print("FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 6))
	print("FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 6))
	print("FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 6))
	print("Verification EER : ", np.round(res.eer * 100, 6))

def find_scores(i1, i2):
	scores, scores_vf = [], []
	fpath = os.path.join(args.mated_min_dir, i1+'__'+i2+'.txt')

	tsp, fsp = [], []

	s1 = i1.split('_')[0]
	n1 = np.load(os.path.join(args.des_dir, i1+'.npy'), allow_pickle=True, encoding='bytes').item()
	m1 = n1[b'mnt'][:, :3]
	d1 = n1[b'des'][0]

	s2 = i2.split('_')[0]
	n2 = np.load(os.path.join(args.des_dir, i2+'.npy'), allow_pickle=True, encoding='bytes').item()
	m2 = n2[b'mnt'][:, :3]
	d2 = n2[b'des'][0]

	our_score = 0
	num_mated = 0
	with open(fpath, 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		for k, row in enumerate(reader):
			if k == 2:
				vf_score = float(row[0])
			elif k == 3:
				num_mated = float(row[0])
			elif k >= 4:
				r1, r2 = -1, -1
				q1 = np.array(row[:3]).astype(np.float32)
				q1[2] = 2*np.pi - (q1[2]*np.pi/180)

				for i, r in enumerate(m1):
					if np.array_equal(r, q1):
						r1 = i
						break

				q2 = np.array(row[3:]).astype(np.float32)
				q2[2] = 2*np.pi - (q2[2]*np.pi/180)
				for i, r in enumerate(m2):
					if np.array_equal(r, q2):
						r2 = i
						break

				assert(r1 != -1 and r2 != -1)
				# if i1 == '99_1' and i2 == '100_1':
					# print(get_similarity(d1[r1], d2[r2]), vf_score)

				if s1 == s2:
					print("True:", i1, i2)
					tsp.append(get_similarity(d1[r1], d2[r2]))
				else:
					print("False:", i1, i2)
					fsp.append(get_similarity(d1[r1], d2[r2]))
				our_score += get_similarity(d1[r1], d2[r2])
	# our_score /= (num_mated + 1e-8)

	if len(tsp):
		assert(len(fsp) == 0)
	if len(fsp):
		assert(len(tsp) == 0)

	return our_score, vf_score, tsp, fsp



true_scores, false_scores = [], []
true_scores_vf, false_scores_vf = [], []
tsp, fsp = [], []

people = args.subjects
fingers = args.impressions

for p in range(1, people + 1):
	for f_1 in range(1, fingers + 1):
		for f_2 in range(f_1 + 1, fingers + 1):
			i1 = str(p)+'_'+str(f_1)
			i2 = str(p)+'_'+str(f_2)
			s, s_vf, tsp_cur, _ = find_scores(i1, i2)
			true_scores.append(s)
			true_scores_vf.append(s_vf)
			tsp.extend(tsp_cur)

for p_1 in range(1, people + 1):
	for p_2 in range(p_1 + 1, people + 1):
		i1 = str(p_1)+'_1'
		i2 = str(p_2)+'_1'
		s, s_vf, _, fsp_cur = find_scores(i1, i2)
		false_scores.append(s)
		false_scores_vf.append(s_vf)
		fsp.extend(fsp_cur)


true_scores = np.array(true_scores)
false_scores = np.array(false_scores)
true_scores_vf = np.array(true_scores_vf)
false_scores_vf = np.array(false_scores_vf)

true_scores_n, false_scores_n = normalize_scores(true_scores, false_scores)
true_scores_vf_n, false_scores_vf_n = normalize_scores(true_scores_vf, false_scores_vf)

calculate_metrics(true_scores, false_scores, "Joint")
calculate_metrics(true_scores_vf, false_scores_vf, "Verifinger")
calculate_metrics(true_scores_vf_n * 0.8 + true_scores_n * 0.2, false_scores_vf_n * 0.8 + false_scores_n * 0.2, "Verifinger 0.8 Joint 0.2")
calculate_metrics(true_scores_vf_n * 0.9 + true_scores_n * 0.1, false_scores_vf_n * 0.9 + false_scores_n * 0.1, "Verifinger 0.9 Joint 0.1")

np.save(args.save_prefix+'_true_scores.npy', np.array(true_scores))
np.save(args.save_prefix+'_false_scores.npy', np.array(false_scores))
np.save(args.save_prefix+'_vf_true_scores.npy', np.array(true_scores_vf))
np.save(args.save_prefix+'_vf_false_scores.npy', np.array(false_scores_vf))

np.save(args.save_prefix+'_patches_true_scores.npy', np.array(tsp))
np.save(args.save_prefix+'_patches_false_scores.npy', np.array(fsp))
