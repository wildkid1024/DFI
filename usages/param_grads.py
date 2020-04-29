import sys
sys.path.append('.') 

import torch
import torch.nn as nn
from ModelWrapper import *
from libs.random_fault import *
from libs.quantize import *

def train_test():
    Conf.load()
    net = ModelWrapper(net_name='resnet50', dataset_name='cifar10')
    # net.train()
    # acc = net.verify()
    param_list = net.param_grad()
    for param in param_list:
        print(param.shape)
    fault_model = RandomFault(frac=1/8)
    quantize_model = Quantize()
    layer_scale = [(2,6)] * len(param_list)
    net.weight_quantize(quantize_model, layer_scale)
    net.verify()
    net.weight_inject(fault_model)
    net.verify()

if __name__ == '__main__':
    train_test()
