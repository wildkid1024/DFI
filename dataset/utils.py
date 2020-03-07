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
from libs.config import Conf

def load_data(dataset_name, root_dir='./dataset/', batch_size=32, transform=torchvision.transforms.ToTensor()):
    dataset_name = dataset_name.upper()
    dataset_dict = {'CIFAR10': datasets.CIFAR10,
                    'CIFAR100':datasets.CIFAR100,
                    'FMNIST': datasets.FashionMNIST,
                    'MNIST': datasets.MNIST,
                    'SVHN': datasets.SVHN
                    'IMAGENET':datasets.ImageNet
                    }
    # data_dir = root_dir + dataset_name
    data_dir = root_dir

    if dataset_name == 'IMAGENET':
        traindir = os.path.join(data_dir, 'train')
        testdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform=transformer)
        test_dataset = datasets.ImageFolder(testdir, transform=transformer)
    else if dataset_name == 'SVHN':
        train_dataset = dataset_dict[dataset_name](root=data_dir, split='train', transform=transform)
        test_dataset = dataset_dict[dataset_name](root=data_dir, split='test', transform=transform)
    else:
        train_dataset = dataset_dict[dataset_name](
            root=data_dir, train=True, transform=transform, download=True)
        test_dataset = dataset_dict[dataset_name](
            root=data_dir, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
