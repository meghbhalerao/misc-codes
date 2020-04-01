import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import DataLoader
import os
import random
import scipy
import scipy.ndimage

class CIFAR10_train(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]  
    def one_hot(label,num_classes):
        one_hot = np.zeros((1,num_classes))
        one_hot[1,label-1] = 1
        return one_hot
        
    def __getitem__(self, index):
        image = self.data[index,1:]
        r = image[0:1024]
        r = r.reshape(32,32)
        r = scipy.misc.imresize(r,(224,224))
        r = scipy.ndimage.zoom(r,7)
        g = image[1024:2048]
        g = g.reshape(32,32)
        g = scipy.ndimage.zoom(g,7)
        b = image[2048:3072]
        b = b.reshape(32,32)
        b = scipy.ndimage.zoom(b,7)
        im_stack = np.zeros((3,224,224))
        im_stack[0] = r
        im_stack[1] = g
        im_stack[2] = b
        gt = self.one_hot(self.data[index,0],10)
        sample = {'image': im_stack, 'gt' : gt}
        return sample