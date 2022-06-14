from matplotlib.transforms import Bbox
import copy
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def norm(ts, fs, s1=None, m=None, s2=None, type="none"):
	ts, fs = copy.deepcopy(ts), copy.deepcopy(fs)

	maxx = max(np.max(ts), np.max(fs))
	minn = min(np.min(ts), np.min(fs))
	print("pre", minn, maxx)

	if type == "minmax":
		# min-max
		ts = (ts - minn) / (maxx - minn)
		fs = (fs - minn) / (maxx - minn)

	elif type == "z":
		# z - score
		mean = np.mean(np.concatenate((ts, fs)))
		std = np.std(np.concatenate((ts, fs)))
		ts = (ts - mean) / std
		fs = (fs - mean) / std

	elif type == "mad":
		# median - MAD
		median = np.median(np.concatenate((ts, fs)))
		ts_m = np.abs(ts - median)
		fs_m = np.abs(fs - median)
		mad = np.median(np.concatenate((ts_m, fs_m)))
		ts = (ts - median) / mad
		fs = (fs - median) / mad

	elif type == "tanh":
		# tanh
		mean = np.mean(np.concatenate((ts, fs)))
		std = np.std(np.concatenate((ts, fs)))
		ts = 0.5 * (np.tanh(0.01 * ((ts - mean) / std)) + 1)
		fs = 0.5 * (np.tanh(0.01 * ((fs - mean) / std)) + 1)	

	elif type == "bisigm":
		# bisigm
		inds = np.argwhere(ts < m)
		ts[inds] = 1 / (1 + np.exp(-2 * (ts[inds] - m) / s1))
		inds = np.argwhere(ts >= m)
		ts[inds] = 1 / (1 + np.exp(-2 * (ts[inds] - m) / s2))
		inds = np.argwhere(fs < m)
		fs[inds] = 1 / (1 + np.exp(-2 * (fs[inds] - m) / s1))
		inds = np.argwhere(fs >= m)
		fs[inds] = 1 / (1 + np.exp(-2 * (fs[inds] - m) / s2))

	maxx = max(np.max(ts), np.max(fs))
	minn = min(np.min(ts), np.min(fs))
	print("post", minn, maxx)

	return ts, fs

def sigmoid(x, k=1, c=0.0):
  
    z = np.exp(-k*(x - c))
    sig = 1 / (1 + z)

    return sig


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--global_dir')
parser.add_argument('--afis_dir')
parser.add_argument('--save_dir')
parser.add_argument('--ts_thresh', type=float, default=1)
parser.add_argument('--fs_thresh', type=float, default=0)
parser.add_argument('--clipping', type=float, default=0)
parser.add_argument('--fuse', default='mean')
parser.add_argument('--norm', default='none')
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()

ts_thresh = args.ts_thresh
fs_thresh = args.fs_thresh

global_ts = np.loadtxt(os.path.join(args.global_dir, 'true_scores_normalized'))
global_fs = np.loadtxt(os.path.join(args.global_dir, 'false_scores_normalized'))
global_ts, global_fs = norm(global_ts, global_fs, 0.3, 0.8, 0.15, type=args.norm)
if args.vis:
	plt.rcParams["figure.figsize"] = (8,5)
	plt.figure()
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.hist(global_ts, bins = np.arange(-0.1, 1.1, 0.01), color='limegreen', alpha=0.7)
	plt.hist(global_fs, bins = np.arange(-0.1, 1.1, 0.01), color='tomato', alpha=0.7)
	# plt.plot(X1[1:], F1)
	plt.yscale('log')
	plt.tight_layout()
	# plt.title('Global scores')
	# plt.show()
	plt.savefig('scores_global')

afis_ts = np.loadtxt(os.path.join(args.afis_dir, 'true_scores_normalized'))
afis_fs = np.loadtxt(os.path.join(args.afis_dir, 'false_scores_normalized'))
afis_ts, afis_fs = norm(afis_ts, afis_fs, 2.5, 3, 2.5, type=args.norm)
if args.vis:
	plt.rcParams["figure.figsize"] = (8,5)
	plt.figure()
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.hist(afis_ts, bins = np.arange(-0.1, 1.1, 0.01), color='limegreen', alpha=0.7)
	plt.hist(afis_fs, bins = np.arange(-0.1, 1.1, 0.01), color='tomato', alpha=0.7)
	# plt.plot(X1[1:], F1)
	plt.yscale('log')
	plt.tight_layout()
	# plt.title('Local scores')
	# plt.show()
	plt.savefig('scores_local')

masked_avg_ts = copy.deepcopy(global_ts)
masked_avg_fs = copy.deepcopy(global_fs)
if args.clipping == 0:
	masked_avg_ts[masked_avg_ts > args.ts_thresh] = (masked_avg_ts[masked_avg_ts > args.ts_thresh] + 1)/2
	masked_avg_ts[masked_avg_ts < args.fs_thresh] = (masked_avg_ts[masked_avg_ts < args.fs_thresh] + 0)/2
else:
	masked_avg_ts[masked_avg_ts > args.ts_thresh] = masked_avg_ts[masked_avg_ts > args.ts_thresh] ** (1/args.clipping) 
	masked_avg_ts[masked_avg_ts < args.fs_thresh] = masked_avg_ts[masked_avg_ts < args.fs_thresh] ** (args.clipping) 
mask_ts = np.argwhere(np.logical_and(masked_avg_ts <= args.ts_thresh, masked_avg_ts >= args.fs_thresh))
notmask_ts = np.argwhere(np.logical_not(np.logical_and(masked_avg_ts <= args.ts_thresh, masked_avg_ts >= args.fs_thresh)))
if args.clipping == 0:
	masked_avg_fs[masked_avg_fs > args.ts_thresh] = (masked_avg_fs[masked_avg_fs > args.ts_thresh] + 1)/2
	masked_avg_fs[masked_avg_fs < args.fs_thresh] = (masked_avg_fs[masked_avg_fs < args.fs_thresh] + 0)/2
else:
	masked_avg_fs[masked_avg_fs > args.ts_thresh] = masked_avg_fs[masked_avg_fs > args.ts_thresh] ** (1/args.clipping) 
	masked_avg_fs[masked_avg_fs < args.fs_thresh] = masked_avg_fs[masked_avg_fs < args.fs_thresh] ** (args.clipping) 
mask_fs = np.argwhere(np.logical_and(masked_avg_fs <= args.ts_thresh, masked_avg_fs >= args.fs_thresh))
notmask_fs = np.argwhere(np.logical_not(np.logical_and(masked_avg_fs <= args.ts_thresh, masked_avg_fs >= args.fs_thresh)))

selected_global_ts = global_ts[mask_ts]
selected_global_fs = global_fs[mask_fs]
if args.vis:
	plt.rcParams["figure.figsize"] = (8,5)
	plt.figure()
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.hist(selected_global_ts, bins = np.arange(-0.1, 1.1, 0.01), color='limegreen', alpha=0.7)
	plt.hist(selected_global_fs, bins = np.arange(-0.1, 1.1, 0.01), color='tomato', alpha=0.7)
	# plt.title('Selected global scores')
	plt.yscale('log')
	plt.tight_layout()
	# plt.show()
	plt.savefig('scores_selected_global')

selected_afis_ts = afis_ts[mask_ts]
selected_afis_fs = afis_fs[mask_fs]
if args.vis:
	plt.rcParams["figure.figsize"] = (8,5)
	plt.figure()
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.hist(selected_afis_ts, bins = np.arange(-0.1, 1.1, 0.01), color='limegreen', alpha=0.7)
	plt.hist(selected_afis_fs, bins = np.arange(-0.1, 1.1, 0.01), color='tomato', alpha=0.7)
	# plt.title('Selected local scores')
	plt.yscale('log')
	plt.tight_layout()
	# plt.show()
	plt.savefig('scores_selected_local')

#print(np.max(np.concatenate((global_ts[mask_ts].reshape(-1, 1), afis_ts[mask_ts].reshape(-1, 1)), axis=1), axis=1).reshape(-1, 1).shape)
#print(masked_avg_ts[mask_ts].shape)
if args.fuse == "max":
	masked_avg_ts[mask_ts] = np.max(np.concatenate((global_ts[mask_ts].reshape(-1, 1), afis_ts[mask_ts].reshape(-1, 1)), axis=1), axis=1).reshape(-1, 1) #(global_ts[mask_ts] + afis_ts[mask_ts]) / 2
	masked_avg_fs[mask_fs] = np.max(np.concatenate((global_fs[mask_fs].reshape(-1, 1), afis_fs[mask_fs].reshape(-1, 1)), axis=1), axis=1).reshape(-1, 1) #(global_fs[mask_fs] + afis_fs[mask_fs]) / 2
elif args.fuse == "min":
	masked_avg_ts[mask_ts] = np.min(np.concatenate((global_ts[mask_ts].reshape(-1, 1), afis_ts[mask_ts].reshape(-1, 1)), axis=1), axis=1).reshape(-1, 1) #(global_ts[mask_ts] + afis_ts[mask_ts]) / 2
	masked_avg_fs[mask_fs] = np.min(np.concatenate((global_fs[mask_fs].reshape(-1, 1), afis_fs[mask_fs].reshape(-1, 1)), axis=1), axis=1).reshape(-1, 1) #(global_fs[mask_fs] + afis_fs[mask_fs]) / 2
elif args.fuse == "mean":
	masked_avg_ts[mask_ts] = (global_ts[mask_ts] + afis_ts[mask_ts]) / 2
	masked_avg_fs[mask_fs] = (global_fs[mask_fs] + afis_fs[mask_fs]) / 2
if args.vis:
	# pass
	plt.rcParams["figure.figsize"] = (8,5)
	plt.figure()
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.hist(masked_avg_ts, bins = np.arange(-0.1, 1.1, 0.01), color='limegreen', alpha=0.7)
	plt.hist(masked_avg_fs, bins = np.arange(-0.1, 1.1, 0.01), color='tomato', alpha=0.7)
	# plt.title('Scores masked avg')
	plt.yscale('log')
	plt.tight_layout()
	# plt.show()
	plt.savefig('scores_masked_avg')

selected_masked_avg_ts = masked_avg_ts[mask_ts]
selected_masked_avg_fs = masked_avg_fs[mask_fs]

notselected_masked_avg_ts = masked_avg_ts[notmask_ts]
notselected_masked_avg_fs = masked_avg_fs[notmask_fs]

num_afis = mask_ts.shape[0] + mask_fs.shape[0]
num_tot = global_ts.shape[0] + global_fs.shape[0]
print("% pairs needing afis scores: ", 100 * num_afis / num_tot)
print('Avg masked avg true score: ', np.mean(masked_avg_ts))
print('Avg masked avg false score: ', np.mean(masked_avg_fs))


if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)
np.savetxt(os.path.join(args.save_dir, 'true_scores_normalized'), masked_avg_ts)
np.savetxt(os.path.join(args.save_dir, 'false_scores_normalized'), masked_avg_fs)
print(masked_avg_ts.shape, masked_avg_fs.shape)
res = get_eer_stats(masked_avg_ts, masked_avg_fs)
print("FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 2))
print("FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 2))
print("FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 2))
print("Verification EER : ", np.round(res.eer * 100, 2))


