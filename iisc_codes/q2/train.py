import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import time 
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from data import CIFAR10_train
from data_val import CIFAR10_val
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def normalize(matrix):
    mean = np.mean(matrix.flatten())
    sigma = np.std(matrix.flatten())
    matrix = (matrix - mean)/sigma
    return matrix 
    
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
for row in range(data_val.shape[0]):
    data_val[row,:] = normalize(data_val[row,:])  
end = time.time()
print("Normalizing data took: ", (end-start)/60, " minutes")
# Concatenation of data and labels for easier understanding
data_train = np.concatenate((labels_train,data_train),axis=1)
data_val = np.concatenate((labels_val,data_val),axis=1)
# Setting up the train and validation dataloaders
start  = time.time()
dataset_train = CIFAR10_train(data_train)
train_loader = DataLoader(dataset_train,batch_size= 1,shuffle=True,num_workers=1)
dataset_valid = CIFAR10_val(data_val)
val_loader = DataLoader(dataset_valid, batch_size=1,shuffle=True,num_workers = 1)
end = time.time()
print("Setting up dataloader took: ", (end-start)/60, " minutes")
alexnet = models.alexnet(pretrained=True)
print(alexnet)

print("Training Data Samples: ", len(train_loader.dataset))


optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9,0.999), weight_decay = 0.00005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
############### CHOOSING THE LOSS FUNCTION ###################
loss_fn = x

################ TRAINING THE MODEL##############
for ep in range(num_epochs):
    start = time.time()
    print("\n")
    print("Epoch Started at:", datetime.datetime.now())
    print("Epoch # : ",ep)
    print("Learning rate:", optimizer.param_groups[0]['lr'])
    model.train
    for batch_idx, (subject) in enumerate(train_loader):
        
        # Load the subject and its ground truth
        image = subject['image']
        mask = subject['gt']
        # Loading images into the GPU and ignoring the affine
        image, mask = image.float(), mask.float()
        #Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it   
        image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
        # Making sure that the optimizer has been reset
        optimizer.zero_grad()

        output = model(image.float())
        # Computing the loss
        loss = loss_fn(output.double(), mask.double(),n_classes)
        # Back Propagation for model to learn
        loss.backward()
        #Updating the weight values
        optimizer.step()
        #Pushing the dice to the cpu and only taking its value
        
        scheduler.step()

    # Now we enter the evaluation/validation part of the epoch    
    model.eval        
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            image = subject['image']
            mask = subject['gt']
            image, mask = image.to(device), mask.to(device)
            output = model(image.float())
            





























