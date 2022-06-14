from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
import numpy as np
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

    parser.add_argument('--save_path',
                        help="path to save tf model")

    args = parser.parse_args()
    return args

args = parse_args()
update_config(config, args)
model = build_model(config)
model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
logging.info('=> load model file: {}'.format(model_file))
ext = model_file.split('.')[-1]
if ext == 'pth':
    state_dict = torch.load(model_file, map_location="cpu")
    if "checkpoint" in model_file:
        state_dict = state_dict['state_dict']
else:
    raise ValueError("Unknown model file")
device = torch.device("cuda")
model.to(device)

dummy_input = torch.randn(1, 3,384,384, dtype=torch.float).to(device)
torch.onnx.export(model, dummy_input, args.save_path)




