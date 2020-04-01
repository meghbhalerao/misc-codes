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
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

############### CHOOSING THE LOSS FUNCTION ###################
if which_loss == 'dc':
    loss_fn  = MCD_loss
if which_loss == 'dcce':
    loss_fn  = DCCE
if which_loss == 'ce':
    loss_fn = CE
if which_loss == 'mse':
    loss_fn = MCD_MSE_loss
############## STORING THE HISTORY OF THE LOSSES #################
avg_val_loss = 0
total_val_loss = 0
best_val_loss = 2000
best_tr_loss = 2000
total_loss = 0
total_dice = 0
best_idx = 0
best_n_val_list = []
val_avg_loss_list = []
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
        image, mask = image.float().to(device), mask.float().to(device)
        #Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it   
        image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
        # Making sure that the optimizer has been reset
        optimizer.zero_grad()
        # Forward Propagation to get the output from the models
        torch.cuda.empty_cache()
        output = model(image.float())
        # Computing the loss
        loss = loss_fn(output.double(), mask.double(),n_classes)
        # Back Propagation for model to learn
        loss.backward()
        #Updating the weight values
        optimizer.step()
        #Pushing the dice to the cpu and only taking its value
        curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
        #train_loss_list.append(loss.cpu().data.item())
        total_loss+=curr_loss
        # Computing the average loss
        average_loss = total_loss/(batch_idx + 1)
        #Computing the dice score 
        curr_dice = 1 - curr_loss
        #Computing the total dice
        total_dice+= curr_dice
        #Computing the average dice
        average_dice = total_dice/(batch_idx + 1)
        scheduler.step()
        torch.cuda.empty_cache()
    print("Epoch Training dice:" , average_dice)      
    if average_dice > 1-best_tr_loss:
        best_tr_idx = ep
        best_tr_loss = 1 - average_dice
    total_dice = 0
    total_loss = 0     
    print("Best Training Dice:", 1-best_tr_loss)
    print("Best Training Epoch:", best_tr_idx)
    # Now we enter the evaluation/validation part of the epoch    
    model.eval        
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            image = subject['image']
            mask = subject['gt']
            image, mask = image.to(device), mask.to(device)
            output = model(image.float())
            curr_loss = dice_loss(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()).cpu().data.item()
            total_loss+=curr_loss
            # Computing the average loss
            average_loss = total_loss/(batch_idx + 1)
            #Computing the dice score 
            curr_dice = 1 - curr_loss
            #Computing the total dice
            total_dice+= curr_dice
            #Computing the average dice
            average_dice = total_dice/(batch_idx + 1)

    print("Epoch Validation Dice: ", average_dice)
    torch.save(model, model_path + which_model  + str(ep) + ".pt")
    if ep > save_best:
        keep_list = np.argsort(np.array(val_avg_loss_list))
        keep_list = keep_list[0:save_best]
        for j in range(ep):
            if j not in keep_list:
                if os.path.isfile(os.path.join(model_path + which_model  + str(j) + ".pt")):
                    os.remove(os.path.join(model_path + which_model  + str(j) + ".pt"))
        
        print("Best ",save_best," validation epochs:", keep_list)

    total_dice = 0
    total_loss = 0
    stop = time.time()   
    val_avg_loss_list.append(1-average_dice)  
    print("Time for epoch:",(stop - start)/60,"mins")    
    sys.stdout.flush()





























