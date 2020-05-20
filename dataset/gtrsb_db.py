from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs

import matplotlib.pyplot as plt
import csv

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath, train=True):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels

    def readGT(gtFilePath=''):
        gtFile = open(gtFilePath)
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader) # skip header
        # loop over all images in current annotations file
        i = 0 
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[-1]) # the 8th column is the label
        gtFile.close()
        return images, labels
    
    if not train:
        prefix = rootpath + "Images/"
        gtFilePath =  rootpath + 'GT-final_test.csv'
        readGT(gtFilePath=gtFilePath)
        return images, labels
    
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFilePath = prefix + 'GT-'+ format(c, '05d') + '.csv'
        readGT(gtFilePath=gtFilePath)
    return images, labels

class GTSRB(data.Dataset):
    '''
    Traffic Dataset.
    '''

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = readTrafficSigns(root)
        else:
            self.test_data, self.test_labels = readTrafficSigns(root, train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        img = Image.fromarray(img)
        # img = img.astype(np.float32)
        # img = np.resize(img, (48,48,3) )
        # print(img.shape)
        # print(target)
        target = torch.Tensor([int(target)])[0]
        target = target.long()
        # print(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

