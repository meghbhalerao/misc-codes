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
data_test = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/data_test.txt")
labels_test = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/labels_test.txt")
end = time.time()
print("Loading data took: ", (end-start)/60, " minutes")

# Expanding dimentions of labels for concatenation with image data
labels_train_full = np.expand_dims(labels_train_full,axis=1)
labels_test = np.expand_dims(labels_test,axis=1)
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



dataset_train = CIFAR10(,psize)
train_loader = DataLoader(dataset_train,batch_size= batch,shuffle=True,num_workers=1)
dataset_valid = TumorSegmentationDataset_val(df_final_val,psize)
val_loader = DataLoader(dataset_valid, batch_size=1,shuffle=True,num_workers = 1)