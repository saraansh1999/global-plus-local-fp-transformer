import time
from pyeer.eer_info import get_eer_stats
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
import argparse
import torch

def get_metric(embs_dict, people, accq, score_path=None, people_start=1, accq_start=1):
	# true scores
	print("Calculating True Scores: ")
	ts = []
	for p in tqdm(range(people_start, people_start + people)):
		for f_1 in range(accq_start, accq + accq_start):
			for f_2 in range(f_1 + 1, accq + accq_start):
				k_1 = str(p) + '_' + str(f_1)
				k_2 = str(p) + '_' + str(f_2)
				try:
					score = (embs_dict[k_1].reshape(1, -1) @ embs_dict[k_2].reshape(1, -1).T).item()
					ts.append(score)
				except Exception as e:
					print(e, "SKIPPED:", k_1, k_2)

	# false scores
	print("Calculating False Scores: ")
	fs = []
	for p_1 in tqdm(range(people_start, people_start + people)):
		for p_2 in range(p_1 + 1, people_start + people):
			k_1 = str(p_1) + '_' + str(1)
			k_2 = str(p_2) + '_' + str(1)
			try:
				score = (embs_dict[k_1].reshape(1, -1) @  embs_dict[k_2].reshape(1, -1).T).item()
				fs.append(score)
			except Exception as e:
				print(e, "SKIPPED:", k_1, k_2)

	# metrics
	ts, fs = np.array(ts), np.array(fs)
	mini = min(np.min(ts), np.min(fs))
	maxi = max(np.max(ts), np.max(fs))
	tsn = (ts - mini) / (maxi - mini)
	fsn = (fs - mini) / (maxi - mini)
	if score_path != None:
		if not os.path.exists(score_path):
			os.makedirs(score_path)
		np.savetxt(os.path.join(score_path, 'true_scores_normalized'), tsn)
		np.savetxt(os.path.join(score_path, 'false_scores_normalized'), fsn)
	res = get_eer_stats(tsn, fsn)
	print("FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 2))
	print("FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 2))
	print("FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 2))
	print("Verification EER : ", np.round(res.eer * 100, 2))

	return res, tsn, fsn

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--embs_path', help='path of the embs file of format: dict with \
										key as person_accquisition and value as emb')
	parser.add_argument('--people', type=int, help='No of different subjects')
	parser.add_argument('--accq', type=int, help='No of different accquisitions per subject')

	parser.add_argument('--score_path', help='prefix of the path to save the scores in')
	parser.add_argument('--people_start', type=int, default=1)
	parser.add_argument('--accq_start', type=int, default=1)
	args = parser.parse_args()

	embs_dict = np.load(args.embs_path, allow_pickle=True).item()

	get_metric(embs_dict, args.people, args.accq, args.score_path, args.people_start, args.accq_start)
