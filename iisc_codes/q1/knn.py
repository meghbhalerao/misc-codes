import numpy as np
import pickle
import time 

something = 0
# =============================================================================
# def get_k(data_val,data_train,K):
#     for k in range(1,K):
#         
#     return something
# =============================================================================

def L2_distance():
    
    return something

def cosine_distance():
    return something

def normalize(matrix):
    mean = np.mean(matrix.flatten())
    sigma = np.std(matrix.flatten())
    matrix = (matrix - mean)/sigma
    return matrix 
    

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
# Concatenation of the labels with the data
for row in range(data_train_full.shape[0]):
    data_train_full[row,:] = normalize(data_train_full[row,:])    
data_train_full = np.concatenate((labels_train_full,data_train_full),axis=1)
data_test = np.concatenate((labels_test,data_test),axis=1)
# Before this a random shuffling of the training data can be done once I figure out why the np shuffle funcrtion is not working
# Splitting the training data further into training and validation data
data_train = data_train_full[0:45000,:]
data_val = data_train_full[45000:50000,:]
# Preprocessing the training, validation and test data