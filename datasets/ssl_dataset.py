import megengine as mge
from megengine.data import transform

from .data_utils import split_ssl_data
from .dataset import BasicDataset
import numpy as np
import json
import os

import random
import sys
import copy


def get_transform(crop_size, train=True):
    if train:
        return transform.Compose([transform.RandomHorizontalFlip(),
                                transform.RandomCrop(crop_size, padding_size=4)]) #padding_mode='reflect'])
    return None


class SSL_Dataset:
    def __init__(self, args, name='cifar10', train=True, num_classes=10, data_dir='./data'):
        self.args = args
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        self.transform = get_transform(crop_size, train)

    def get_data(self, svhn_extra=True):
        if self.name.upper() in ['SVHN', 'STL10']:
            print('megengine dataset not supports these datasets, we will add these datasets in Pytorch code. Please waitting...')
            exit()
        dset = getattr(mge.data.dataset, self.name.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets = dset.arrays[:2]
        data = np.stack(data, axis=0)[:,:,:,::-1] # BGR -> RGB
        return data, targets
    
    def get_dset(self, is_ulb=False, strong_transform=None):

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform

        return BasicDataset(self.name, data, targets, num_classes, transform, is_ulb, strong_transform)


    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True, strong_transform=None):

        data, targets = self.get_data()
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
                                                                    num_labels, self.num_classes,
                                                                    index, include_lb_to_ulb)

        lb_dset = BasicDataset(self.name, lb_data, lb_targets, self.num_classes, self.transform, False, None)
        ulb_dset = BasicDataset(self.name, ulb_data, ulb_targets, self.num_classes, self.transform, True, strong_transform)
        return lb_dset, ulb_dset
