import argparse, sys, os
sys.path.insert(0, sys.path[0] + "/../")
current_dir = os.path.dirname(sys.argv[0])
from utils import *
import torch
import numpy as np
import random

# Parameter Setting
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# args
args = argparse.ArgumentParser()
input_root_dir = check_system()  # todo modify

# defaults
args.add_argument('--data_size', default=224, help='data size')
args.add_argument('--num_classes', default=2)
args.add_argument('--channel_ratio', default=4)
args.add_argument('--cv_nums', default=3)
args.add_argument('--input_dir', default=input_root_dir)
args.add_argument('--seed', default=0)
args.add_argument('--device', default='cuda:0')