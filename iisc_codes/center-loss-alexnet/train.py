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
import copy


num_classes = 10
lam = 0.1
feat_dim = 4096
batch_size = 20
alpha = 0.3
df_train_full = pd.read_csv("train.csv")


# Defining the train and validation sets clearly
data_train = df_train_full[0:45000]
data_val = df_train_full[45000:50000]


# Setting up the dataloader
dataset_train = CIFAR10_train(data_train)
train_loader = DataLoader(dataset_train,batch_size= batch_size,shuffle=True,num_workers=4)
dataset_valid = CIFAR10_val(data_val)
val_loader = DataLoader(dataset_valid, batch_size=batch_size,shuffle=True,num_workers = 4)


# Defining a pre-existing model in torch
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(feat_dim,num_classes)


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

# Initializing the centers matrix by random numbers
center_matrix = torch.randn(feat_dim, num_classes)

# Setting up some loss functions
def MSE_loss(mask,gt):
    diff = mask.flatten() - gt.flatten()
    loss = sum(diff*diff)
    return loss

def Center_loss(deep_features_batch, mask_batch, class_center):   
    loss = 0
    for sample in range(deep_features_batch.shape[1]):
        cl = np.argmax(mask_batch[:,sample])
        loss+=MSE_loss(deep_features_batch[:,sample],class_center[:,cl])
    return loss




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
        image = subject['image'].float()
        mask = subject['gt'].float()
        

        # Setting the optimizer to zero
        optimizer.zero_grad()
        
        
        # Forward passing the image through both the networks - standard net and feature net 
        output = model(image.float())
        feature = feature_model(image.float())
        # The matrix which is used to update the cluster centers
        delta_center_matrix = torch.zeros([feat_dim, num_classes], dtype=torch.float)

        # Computing and updating the cluster centers according to the present mini-batch 
        for cl in range(1,num_classes+1):
            for sample in range(batch_size):
                if np.argmax(mask.T[:,sample].numpy()) == cl:
                    delta_center_matrix[:,cl]+= feature.T[:,sample]
            
        center_matrix = center_matrix + alpha*delta_center_matrix   
        
        
        

        feature = feature.T
        # Computing the loss as a combination of the center loss and the MSE Loss
        loss = Center_loss(feature.double(),mask.T.double(),center_matrix.double())*lam + MSE_loss(output.double(), mask.double())
        
        
        # Backpropagation of the loss
        loss.backward()
        
        
        #Updating the weight values
        optimizer.step()
        
        #Pushing the ground truth and predicted class to the cpu to calculate the accuracy for particular epoch
        mask = mask.cpu().detach().numpy().T
        output = output.cpu().detach().numpy().T
        temp = np.zeros_like(output)
        
        # Loop for getting one hot of the output array
        col = 0
        for i in np.argmax(output,axis=0):
            temp[i,col] = 1
            col =  col+1
        
        train_acc+=sum(sum(mask*temp))
        train_loss+=loss.cpu().detach().numpy()

    # Displaying loss and accuracy after the epoch 
    print("Training Accuracy for epoch # ", ep, " is: ",train_acc/((batch_idx+1)*batch_size))
    print("Training Loss for epoch # ", ep, " is: ",train_loss/((batch_idx+1)*batch_size))

    
    # Resetting the loss and accuracy after each epoch
    train_acc = 0
    train_loss = 0
    
    
    # Now we enter the evaluation/validation part of the epoch - this is same with or without the center loss 
    model.eval        
     
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            # Load the subject and its ground truth
            image = subject['image'].float()
            mask = subject['gt'].float()
            
            
            # Forward passing the image through both the networks - standard net and feature net 
            output = model(image.float())
            
            #Pushing the ground truth and predicted class to the cpu to calculate the accuracy for particular epoch
            mask = mask.cpu().detach().numpy().T
            output = output.cpu().detach().numpy().T
            temp = np.zeros_like(output)
            
            # Loop for getting one hot of the output array
            col = 0
            for i in np.argmax(output,axis=0):
                temp[i,col] = 1
                col =  col+1
            val_acc+=sum(sum(mask*temp))
            val_loss+=loss.cpu().detach().numpy()

    # Displaying loss and accuracy after the epoch 
    print("Validation Accuracy for epoch # ", ep, " is: ",val_acc/((batch_idx+1)*batch_size))
    print("Validation Loss for epoch # ", ep, " is: ",val_loss/((batch_idx+1)*batch_size))

    scheduler.step(val_acc)
    
    
    val_acc = 0
    val_loss = 0
    
    
    end = time.time()
    print("Time taken for this epoch is: ", (end-start)/60, "minutes")

            
            





























