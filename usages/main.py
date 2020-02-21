#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/8 21:00
# @Author  : wildkid1024
# @Site    : 
# @File    : main2.py
# @Software: PyCharm
import time

import numpy as np
import torch
import torch.nn as nn

from ModelWrapper import *

from functions import get_para


def train_test():
    Conf.load()
    net = ModelWrapper()
    net.train()
    _, acc = net.verify()
    print(acc)


def get_weights_grad():
    config = get_para()
    net = ModelWrapper(cfg=config)
    net.verify()
    weight = net.param_grad()
    print("Get the {} images weight!".format(len(weight)))
    torch.save(weight, './results/LeNet_FMNIST_conv.weight.grad')
    print(weight[0]['features.0.weight'].size())


def get_weights():
    config = get_para()
    net = ModelWrapper(cfg=config)
    net.verify()
    net.weights_export()


def summary():
    config = get_para()
    net = ModelWrapper(cfg=config)
    net.verify()
    net.summary()


if __name__ == '__main__':
    train_test()
