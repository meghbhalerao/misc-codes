import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import time 
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models


start = time.time()
data_train_full = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/data_train.txt")
labels_train_full = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/labels_train.txt")
end = time.time()
print("Loading data took: ", (end-start)/60, " minutes")

# Expanding dimentions of labels for concatenation with image data
labels_train_full = np.expand_dims(labels_train_full,axis=1)
# Defining the train and validation sets clearly
data_train = data_train_full[0:45000,:]
data_val = data_train_full[45000:50000,:]
labels_train = labels_train_full[0:45000,:]
labels_val = labels_train_full[45000:50000,:]
# Normalization of the data elements
start = time.time()
for row in range(data_train.shape[0]):
    data_train[row,:] = normalize(data_train[row,:])  
for row in range(data_test.shape[0]):
    data_test[row,:] = normalize(data_test[row,:])  
for row in range(data_val.shape[0]):
    data_val[row,:] = normalize(data_val[row,:])  
end = time.time()
print("Normalizing data took: ", (end-start)/60, " minutes")
# Concatenation of data and labels for easier understanding
data_train = np.concatenate((labels_train,data_train))
data_val = np.concatenate((labels_val,data_val))
# Setting up the train and validation dataloaders
dataset_train = CIFAR10_train(data_train)
train_loader = DataLoader(dataset_train,batch_size= 1,shuffle=True,num_workers=1)
dataset_valid = CIFAR10_val(data_val)
val_loader = DataLoader(dataset_valid, batch_size=1,shuffle=True,num_workers = 1)