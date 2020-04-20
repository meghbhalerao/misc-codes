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
import pandas as pd
import datetime
from center_loss import CenterLoss
import copy

def MSE_loss(mat1,mat2):
    diff = mat1.flatten() - mat2.flatten()
    loss = sum(diff*diff)
    return loss


num_classes = 10
df_train_full = pd.read_csv("train.csv")


# Defining the train and validation sets clearly
data_train = df_train_full[0:45000]
data_val = df_train_full[45000:50000]


# Setting up the dataloader
dataset_train = CIFAR10_train(data_train)
train_loader = DataLoader(dataset_train,batch_size= 1,shuffle=True,num_workers=4)
dataset_valid = CIFAR10_val(data_val)
val_loader = DataLoader(dataset_valid, batch_size=1,shuffle=True,num_workers = 4)


# Defining a pre-existing model in torch
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096,num_classes)


# Freezing the convolutional layer weights
for param in model.features.parameters():
    param.requires_grad = False
    
    
# Freezing the classifier layer weights except the last layer
for n in range(6):
    for param in model.classifier[n].parameters():
        param.requires_grad = False
     
        
# This is the model which gives us the deep features - a 4096 dimensional tensor - this is just done by removing the last layer of the existing alexnet model 
feature_model = copy.deepcopy(model)
del(feature_model.classifier[6])


print("Training Data Samples: ", len(train_loader))


optimizer = optim.Adam(model.classifier[6].parameters(), lr = 0.1, betas = (0.9,0.999), weight_decay = 0.00005)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=False, threshold=0.003, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

###### PARAMETERS NEEDED TO BE MONITORED ##########
train_acc = 0
val_acc = 0
train_loss = 0
val_loss = 0
num_epochs = 100
alpha = 0.1

################ TRAINING THE MODEL ##############
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
        optimizer.zero_grad()
        output = model(image.float())
        feature = feature_model(image.float())
        
        # Computing the loss
        mask = mask.T
        # Combination of the center loss and standard MSE loss
        loss = center_loss(feature.double(), mask[0,:].double())*alpha + MSE_loss(output.double(), mask.double())
        # Backpropagation of the loss
        loss.backward()
        # Updating the parameters of the center loss
        for param in center_loss.parameters():
            param.grad.data *= (1./alpha)
       
        # Updating the parameters of the center loss as well as the network
        optimizer_centloss.step()
        optimizer_centloss.zero_grad()
        
        #Updating the weight values
        optimizer.step()
        
        #Pushing the ground truth and predicted class to the cpu
        mask = mask.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        temp = np.zeros((1,num_classes))
        temp[0,np.argmax(output)] =  1
        train_acc+=sum(mask*temp)
        train_loss+=loss.cpu().detach().numpy()
        
    print("Training Accuracy for epoch # ", ep, " is: ",train_acc/(batch_idx+1))
    print("Training Loss for epoch # ", ep, " is: ",train_loss/(batch_idx+1))

    train_acc = 0
    train_loss = 0
    # Now we enter the evaluation/validation part of the epoch - this is same with or without the center loss 
    model.eval        
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            image = subject['image']
            mask = subject['gt']
            image, mask = image.to(device), mask.to(device)
            output = model(image.float())
            mask = mask.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            loss = loss_fn(output.double(), mask.double())
            temp = np.zeros((1,num_classes))
            temp[0,np.argmax(output)] = 1     
            val_acc+=sum(mask*temp)
            val_loss+=loss.cpu().detach().numpy()
    print("Validation Accuracy for epoch # ",ep," is: ",val_acc/(batch_idx+1))
    print("Validation Loss for epoch # ",ep," is: ",val_loss/(batch_idx+1))
    scheduler.step(val_acc)
    val_acc = 0
    val_loss = 0
    end = time.time()
    print("Time taken for this epoch is: ", (end-start)/60, "minutes")

            
            





























