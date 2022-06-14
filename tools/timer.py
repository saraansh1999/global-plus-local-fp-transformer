from __future__ import absolute_import
import time
import numpy as np
import torch
import _init_paths
from models import build_model
import argparse
from config import config
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

args = parse_args()
update_config(config, args)
model = build_model(config)
device = torch.device("cpu")
#device = torch.device("cuda")
model.to(device)
print(model)
dummy_input = torch.randn(1, 3,384,384, dtype=torch.float).to(device)

# INIT LOGGERS
#starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for i in range(10):
    print("warmup", i)
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        print("repetition", rep)
#        starter.record()
        start_time = time.time()
        _ = model(dummy_input)
#        ender.record()
        # WAIT FOR GPU SYNC
#        torch.cuda.synchronize()
        curr_time = time.time() - start_time
#        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)
