import numpy as np
import torch
import argparse
import sys
import os

def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--path', type=str, help='dataset directory', default='./data')
    parser.add_argument('--data', type=str, help='dataset',
                        choices=['bank', 'credit', 'news', 'mushroom', 'nursery', 'student', 'bike', 'covertype'],
                        default='bank')
    # neural network hyper-parameters
    parser.add_argument('--net', type=int, nargs='*', help='number of neurons in each hidden layer', default=[600, 300, 100])
    parser.add_argument('--bs', type=int, help='batch size', default=128)
    parser.add_argument('--seed', type=int, help='random seed', default=1)
    # experiment settings
    parser.add_argument('--repeat', type=int, help='number of trials')
    parser.add_argument('--verbose', action='store_true', help='print train accuracy and loss')
    args = parser.parse_args()

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()
