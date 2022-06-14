#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time
from PIL import Image
import numpy as np
from glob import glob
import os
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import argparse
import sys
import subprocess
from pyeer.eer_info import get_eer_stats

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str)
    parser.add_argument('--people', type=int)
    parser.add_argument('--impressions', type=int)
    parser.add_argument('--savename', type=str)
    parser.add_argument('--people_start', type=int, default=1)
    parser.add_argument('--accq_start', type=int, default=1)
    return parser.parse_args(argv)

def get_scores_afis(p1, p2):
    if not os.path.exists(p1) or not os.path.exists(p2): raise Exception('file not found')
    command = '/home/saraansh.tandon/MSU-LatentAFIS/matching/matcher_topN30_threshold20_threshold20 -c /home/saraansh.tandon/MSU-LatentAFIS/matching/codebook_EmbeddingSize_96_stride_16_subdim_6.dat -f1 ' \
    + p1 + ' -f2 ' + p2
    popen = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    ret = float(output.decode('ascii'))
    return ret

def run_latentafis(path, people, fingers, save, people_start, accq_start):

    (true_scores, false_scores) = ([], [])
    (fps_true, fps_false) = ([], [])
    tsk, fsk = 0, 0
    times = []
    done = 0;
    for p in range(people_start, people_start + people):
        for f_1 in range(accq_start, fingers + accq_start):
            for f_2 in range(f_1 + 1, fingers + accq_start):
                path_1 = os.path.join(path, str(p), str(p) + '_' + str(f_1) + '.dat')
                path_2 = os.path.join(path, str(p), str(p) + '_' + str(f_2) + '.dat')
                try:
#                if 1:
                    start = time.time()
                    score = get_scores_afis(path_1, path_2)
                    times.append(time.time() - start)
                    true_scores.append(score)
                    fps_true.append((path_1, path_2))
#                else:
                except:
                    true_scores.append(1)
                    tsk = tsk + 1
                    print("SKIPPED:", path_1, path_2)
                done = done + 1;
                if done % 100 == 0:
                    print("Done!", done)
                    sys.stdout.flush()

    done = 0;
    for p_1 in range(people_start, people_start + people):
        for p_2 in range(p_1 + 1, people_start + people):
            path_1 = os.path.join(path, str(p_1), str(p_1) + '_' + str(1) + '.dat')
            path_2 = os.path.join(path, str(p_2), str(p_2) + '_' + str(1) + '.dat')
            try:
                score = get_scores_afis(path_1, path_2)
                false_scores.append(score)
                fps_false.append((path_1, path_2))
            except:
                false_scores.append(0)
                fsk = fsk + 1
                print("SKIPPED:", path_1, path_2)
            done = done + 1;
            if done % 100 == 0:
                print("Done!", done)
                sys.stdout.flush()

    ts, fs = np.array(true_scores), np.array(false_scores)

    print(tsk, fsk)

    mini = 0 #min(np.min(ts), np.min(fs))
    maxi = 1 #max(np.max(ts), np.max(fs))
    tsn = (ts - mini) / (maxi - mini)
    fsn = (fs - mini) / (maxi - mini)
    if not os.path.exists(save):
        os.makedirs(save)
    np.savetxt(os.path.join(save, 'true_scores_normalized'), tsn)
    np.savetxt(os.path.join(save, 'false_scores_normalized'), fsn)
    res = get_eer_stats(tsn, fsn)
    print("FRR @ FAR 0.0% : ", np.round(res.fmr0 * 100, 2))
    print("FRR @ FAR 0.1% : ", np.round(res.fmr1000 * 100, 2))
    print("FRR @ FAR 1.0% : ", np.round(res.fmr100 * 100, 2))
    print("Verification EER : ", np.round(res.eer * 100, 2))
    sys.stdout.flush()

    print("Avg time per comparison:", sum(times[50:]) / len(times[50:]))
    return np.array(true_scores), np.array(false_scores), fps_true, fps_false

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    run_latentafis(args.dir, args.people, args.impressions, args.savename, args.people_start, args.accq_start)
