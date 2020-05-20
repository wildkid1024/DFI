#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 10:10
# @Author  : wildkid1024
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from libs.config import Conf
from dataset.gtrsb_db import GTSRB

def load_data(dataset_name, root_dir='./dataset/', n_worker=0, batch_size=4096, transform=transforms.ToTensor()):
    dataset_dict = {'CIFAR10': datasets.CIFAR10,
                    'CIFAR100':datasets.CIFAR100,
                    'FMNIST': datasets.FashionMNIST,
                    'MNIST': datasets.MNIST,
                    'SVHN': datasets.SVHN,
                    }
    data_dir = root_dir + dataset_name.upper()
    # data_dir = root_dir

    if dataset_name.upper() == 'IMAGENET':
        data_dir = root_dir + 'ImageNet/'
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

    elif dataset_name.upper() == 'SVHN':
        train_dataset = dataset_dict[dataset_name](root=data_dir, split='train', transform=transform)
        test_dataset = dataset_dict[dataset_name](root=data_dir, split='test', transform=transform)
    elif dataset_name.upper() == 'GTSRB':
        traindir = os.path.join(data_dir, 'GTSRB/Final_Training/Images/')
        valdir = os.path.join(data_dir, 'GTSRB/Final_Test/')
        transform = transforms.Compose([
            transforms.Scale(48),
            transforms.CenterCrop((48, 48)),
            transforms.ToTensor()
        ])
        train_dataset = GTSRB(
            root=traindir,
            train=True, 
            transform=transform 
        )
        test_dataset = GTSRB(
            root=valdir,
            train=False, 
            transform=transform 
        )
    else:
        dataset_name = dataset_name.upper()
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = dataset_dict[dataset_name](
            root=data_dir, train=True, transform=transform_train, download=True)
        test_dataset = dataset_dict[dataset_name](
            root=data_dir, train=False, transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_worker, pin_memory=True)

    return train_loader, test_loader


def get_split_train_dataset(dataset_name, batch_size, n_worker, val_size, train_size=None, random_seed=1,
                            data_root='data/imagenet', for_inception=False, shuffle=True):
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:
        # use the same order
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('==> Preparing data..')
    if dataset_name == 'imagenet':
        data_root = data_root + 'ImageNet/'
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        valset = datasets.ImageFolder(traindir, test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                num_workers=n_worker, pin_memory=True)
        n_class = 1000
    elif dataset_name == 'gtsrb':
        traindir = os.path.join(data_root, 'GTSRB/Final_Training/Images/')
        valdir = os.path.join(data_root, 'val')
        transform = transforms.Compose([
            transforms.Scale(48),
            transforms.CenterCrop((48, 48)),
            transforms.ToTensor()
        ])
        trainset = GTSRB(
            root=traindir,
            train=True, 
            transform=transform 
        )
        N = int(len(trainset) * 0.7)
        train_db, val_db = torch.utils.data.random_split(trainset, [N, len(trainset)-N])
        train_loader = torch.utils.data.DataLoader(
            train_db,               
            batch_size=batch_size,  
            shuffle=True,            
        )
        val_loader = torch.utils.data.DataLoader(
            val_db,               
            batch_size=batch_size,  
            shuffle=True,   
        )
        n_class = 43
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
