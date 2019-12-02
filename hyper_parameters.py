#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 14:58
# @Author  : wildkid1024
# @Site    : 
# @File    : hyper-parameters.py
# @Software: PyCharm


# 设置网络超参，此为pretrain的参数，neural_sort的参数，neural_pruning的参数，和neural_retrain的部分参数
batch_size = 32
learning_rate = 0.01  # pretrain的学习率
epoches = 50  # pretrain的epoch次数
class_num = 10  # 指出神经网络的分类数量。mnist数据集的输出是0-9

# dataset_root = './dataset/MNIST/'
# pretrained_model = './nn_models/pretrained/lenet_mnist.t2'

# dataset_root = './dataset/FMNIST/'
# pretrained_model = './nn_models/pretrained/lenet_fashionmnist.pkl'

# dataset_root = './dataset/FMNIST/'
pretrained_model = './nn_models/pretrained/test.pkl'

# test_num = 256
inject_times = 3

iteration_num = 256 // batch_size + (256 % batch_size != 0)
