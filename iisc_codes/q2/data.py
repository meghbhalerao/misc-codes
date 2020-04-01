import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import DataLoader
import os
import random
import scipy

class CIFAR10_train(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return data.shape[0]
    def transform(self,img ,gt, dim):
        if random.random()<0.12:
            img = scipy.ndimage.rotate(img,45,axes=(2,1,0),reshape=False,mode='constant')
            gt = scipy.ndimage.rotate(gt,45,axes=(2,1,0),reshape=False,order=0) 
            img, gt = img.copy(), gt.copy() 
        if random.random()<0.12:
            img, gt = np.flipud(img).copy(),np.flipud(gt).copy()
        if random.random() < 0.12:
            img, gt = np.fliplr(img).copy(), np.fliplr(gt).copy()
        if random.random() < 0.12:
            for n in range(dim-1):
                img[n] = gaussian(img[n],True,0,0.1)   
        return img,gt        

    def __getitem__(self, index):
        image = self.data[index,1:]
        r = image[0:1024]
        r = r.reshape(32,32)
        g = image[1024:2048]
        g = g.reshape(32,32)
        b = image[2048:3072]
        b = b.reshape(32,32)
        gt = self.data[index,0]
        sample = {'image': im_stack, 'gt' : gt}
        return sample