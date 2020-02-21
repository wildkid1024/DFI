#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/2 18:17
# @Author  : wildkid1024
# @Site    : 
# @File    : train_cifar10.py
# @Software: PyCharm
import sys
sys.path.append('.') 

import torch
import torch.nn as nn
from ModelWrapper import *
from functions import get_para


def train_test():
    Conf.load()
    net = ModelWrapper(net_name='ResNet18')
    net.train()
    _, acc = net.verify()
    print(acc)


if __name__ == '__main__':
    train_test()
