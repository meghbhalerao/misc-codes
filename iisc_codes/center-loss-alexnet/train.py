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

# Using the mean squared error loss for the network to train
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
train_loader = DataLoader(dataset_train,batch_size=1,shuffle=False,num_workers=4)
dataset_valid = CIFAR10_val(data_val)
val_loader = DataLoader(dataset_valid, batch_size=1,shuffle=False,num_workers = 4)
# Defining a pre-existing model in torch
model = models.alexnet(pretrained=True)
# Modifying the last layer of the model to make the output equal to the number of classes of cifar10
model.classifier[6] = nn.Linear(4096,num_classes)
# Dropping or deleting the last convolutional layer conditionlly
drop = True
if drop == True:
    modules=list(model.children())[0]
    del(modules[10])
# Freezing the convolutional layer weights
for param in model.features.parameters():
    param.requires_grad = False
# Freezing the classifier layer weights except the last layer
for n in range(6):
    for param in model.classifier[n].parameters():
        param.requires_grad = False

print("Training Data Samples: ", len(train_loader))

# Setting up the optimizer
optimizer = optim.Adam(model.classifier[6].parameters(), lr = 0.1, betas = (0.9,0.999), weight_decay = 0.00005)
# Setting a lerning rate scheduler conditioned on the rate of change of loss function and updating the learning rate conditionally after every epoch
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=False, threshold=0.003, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
############### CHOOSING THE LOSS FUNCTION ###################
loss_fn = MSE_loss
###### PARAMETERS NEEDED TO BE MONITORED ##########
train_acc = 0
val_acc = 0
train_loss = 0
val_loss = 0
num_epochs = 100
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
        #Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it   
        #image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
        # Making sure that the optimizer has been reset
        optimizer.zero_grad()
        output = model(image.float())
        # Computing the loss
        loss = loss_fn(output.double(), mask.double())
        # Back Propagation for model to learn
        loss.backward()
        #Updating the weight values
        optimizer.step()
        #Pushing the ground truth and predicted class to the cpu
        mask = mask.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        temp = np.zeros((1,num_classes))
        temp[0,np.argmax(output)] =  1
        train_acc+=sum(sum(mask*temp))
        print(mask,temp)
        train_loss+=loss.cpu().detach().numpy()
    print("Training Accuracy for epoch # ", ep, " is: ",train_acc/(batch_idx+1))
    print("Training Loss for epoch # ", ep, " is: ",train_loss/(batch_idx+1))

    train_acc = 0
    train_loss = 0
    # Now we enter the validation part of the epoch    
    model.eval        
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            image = subject['image']
            mask = subject['gt']
            output = model(image.float())
            mask = mask.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            torch.tensor(mask)
            torch.tensor(output)
            loss = loss_fn(output.double(), mask.double())
            temp = np.zeros((1,num_classes))
            temp[0,np.argmax(output)] = 1     
            val_acc+=sum(sum(mask*temp))
            val_loss+=loss.cpu().detach().numpy()
    print("Validation Accuracy for epoch # ",ep," is: ",val_acc/(batch_idx+1))
    print("Validation Loss for epoch # ",ep," is: ",val_loss/(batch_idx+1))
    scheduler.step(val_acc)
    val_acc = 0
    val_loss = 0
    end = time.time()
    print("Time taken for this epoch is: ", (end-start)/60, "minutes")

            
            





























