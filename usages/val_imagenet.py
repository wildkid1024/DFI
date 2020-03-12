import sys
sys.path.append('.') 

import torch
import torch.nn as nn
from ModelWrapper import *
from functions import get_para


def test_imagenet():
    Conf.load('configs/cfg.yaml')
    net = ModelWrapper(net_name='resnet18', dataset_name='imagenet')
    print(net.model)
    # net.train()
    _, acc = net.verify()
    print(acc)


if __name__ == '__main__':
    test_imagenet()
