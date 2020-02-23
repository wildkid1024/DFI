import sys
sys.path.append('.') 

import torch
import torch.nn as nn
from ModelWrapper import *
from functions import get_para
from libs.random_fault import *

def train_test():
    Conf.load(filename="configs/cfg.yaml")
    net = ModelWrapper(net_name='VGG')
    net.train()
    _, acc = net.verify()
    print(acc)

def vgg_inject():
    Conf.load(filename="configs/cfg.yaml")
    Conf.set("train.resume", True)
    net = ModelWrapper(net_name='VGG',dataset_name='cifar100')
    res1, acc = net.verify()
    print(acc)
    fault_model = RandomFault(frac=1/8)
    net.weight_inject(fault_model)
    res2, acc = net.verify()
    print(acc)
    # print(res1 == res2)

if __name__ == '__main__':
    # train_test()
    vgg_inject()
