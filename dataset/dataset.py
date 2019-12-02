#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 14:38
# @Author  : wildkid1024
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

import collections

import torch
import torchvision
import torchvision.transforms as transforms

from hyper_parameters import batch_size

Dataset = collections.namedtuple('Dataset', 'train_loader,test_loader,classes')

data_root = './'

# -------------------- MNIST dataset --------------------------------------------
mnist_batch_size = batch_size
mnist_root = data_root + 'MNIST/'

train_dataset = torchvision.datasets.MNIST(
    root=mnist_root, train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(
    root=mnist_root, train=False, transform=torchvision.transforms.ToTensor(), download=True)

mnist = Dataset(
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=mnist_batch_size, shuffle=True),
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=mnist_batch_size, shuffle=False),
    classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# -------------------- CIFAR10 dataset ------------------------------------------
cifar10_batch_size = batch_size
cifar10_root = data_root + '/CIFAR/'
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
cifar10_trainset = torchvision.datasets.CIFAR10(root=cifar10_root, train=True, download=False,
                                                transform=cifar10_train_transform)
cifar10_testset = torchvision.datasets.CIFAR10(root=cifar10_root, train=False, download=False,
                                               transform=cifar10_test_transform)
cifar10 = Dataset(
    train_loader=torch.utils.data.DataLoader(cifar10_trainset, batch_size=cifar10_batch_size, shuffle=True,
                                             num_workers=0),
    test_loader=torch.utils.data.DataLoader(
        cifar10_testset, batch_size=cifar10_batch_size, shuffle=False, num_workers=0), classes=cifar10_classes)

# -------------------------------fashion MNIST dataset ---------------------------------------------------------------

fashionmnist_batch_size = batch_size
fashionmnist_root = data_root + '/FMNIST/'

fashionmnist_normalize = torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
fashionmnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))])

fashionmnist_trainset = torchvision.datasets.FashionMNIST(root=fashionmnist_root, train=True,
                                                          transform=fashionmnist_transform,
                                                          download=True)

fashionmnist_testset = torchvision.datasets.FashionMNIST(root=fashionmnist_root,
                                                         train=False,
                                                         transform=fashionmnist_transform,
                                                         download=True)
fashionmnist = Dataset(
    train_loader=torch.utils.data.DataLoader(fashionmnist_trainset, batch_size=fashionmnist_batch_size, shuffle=True),
    test_loader=torch.utils.data.DataLoader(fashionmnist_testset, batch_size=fashionmnist_batch_size, shuffle=False),
    classes=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
)
# -----------------------------------------------------SVHN dataset--------------------------------------------------

svhn_batch_size = batch_size
svhn_root = data_root + '/SVHN/'

svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

svhn_trainset = torchvision.datasets.SVHN(root=svhn_root, split='train', download=False, transform=svhn_transform)
svhn_testset = torchvision.datasets.SVHN(root=svhn_root, split='test', transform=svhn_transform, download=False)

svhn = Dataset(
    train_loader=torch.utils.data.DataLoader(svhn_trainset, batch_size=svhn_batch_size, shuffle=True),
    test_loader=torch.utils.data.DataLoader(svhn_testset, batch_size=svhn_batch_size, shuffle=False),
    classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)