import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import DataLoader
import os
import random
import scipy
import scipy.ndimage as snd
import imageio


class CIFAR10_train(Dataset):
    def __init__(self,df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def one_hot(self,label,num_classes):
        one_hot = np.zeros((1,num_classes))
        one_hot[1,label-1] = 1
        return one_hot
    
    def normalize(self,matrix):
        mean = np.mean(matrix.flatten())
        sigma = np.std(matrix.flatten())
        matrix = (matrix - mean)/sigma
        return matrix 
        
    def __getitem__(self, index):
        image = self.df.iloc[index,0]
        gt = int(self.df.iloc[index,1])
        image = imageio.imread(image)
        image = self.normalize(image)
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]
        r = snd.zoom(r,7)
        g = snd.zoom(g,7)
        b = snd.zoom(b,7)
        im_stack = np.zeros((3,224,224))
        im_stack[0] = r
        im_stack[1] = g
        im_stack[2] = b
        gt = self.one_hot(self.df[index,0],10)
        print(gt)
        print(image)
        subject = {'image': im_stack, 'gt' : gt}
        return subject