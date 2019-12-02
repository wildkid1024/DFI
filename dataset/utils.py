#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 10:10
# @Author  : wildkid1024
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import torch
import torchvision
import torchvision.datasets as datasets


def load_data(dataset_name, batch_size=32, transform=torchvision.transforms.ToTensor()):
    dataset_dict = {'CIFAR10': datasets.CIFAR10,
                    'FMNIST': datasets.FashionMNIST,
                    'MNIST': datasets.MNIST,
                    'SVHN': datasets.SVHN
                    }
    data_dir = './dataset' + dataset_name

    if dataset_name == 'SVHN':
        train_dataset = dataset_dict[dataset_name](root=data_dir, split='train', transform=transform)
        test_dataset = dataset_dict[dataset_name](root=data_dir, split='test', transform=transform)
    else:
        train_dataset = dataset_dict[dataset_name](
            root=data_dir, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
