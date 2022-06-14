import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import argparse
import shutil
import time
import descriptor

#arguements
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', action='store', required=True)
args = parser.parse_args()

model = descriptor.ImportGraph(args.model_dir, input_name="inputs:0", output_name='embedding:0')
dummy_input = np.random.rand(49, 96, 96, 1).astype(np.float32)

# warmup
for i in range(10):
	start = time.time()
	model.run(dummy_input)
	print("Time taken: ", time.time() - start)

times = []
for i in range(100):
	start = time.time()
	model.run(dummy_input)
	times.append(time.time() - start)

print("Avg time", sum(times) / len(times))
