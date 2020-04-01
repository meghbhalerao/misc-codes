import numpy as np
import pickle
import time 

def DM(data_val,data_train):
    dist_mat = np.zeros((data_train.shape[0],data_val.shape[0]))
    for sample_val in range(data_val.shape[0]):
        for sample_train in range(data_train.shape[0]):
            dist_mat[sample_train,sample_val] = L2_distance(data_val[sample_val,1:],data_train[sample_train,1:])         
    return dist_mat

def L2_distance(mat1,mat2):
    dist = np.linalg.norm(mat1-mat2)    
    return dist
def get_most_frequent(array):
    counts = np.bincount(a)
    return np.argmax(counts)

something = 0
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
# Now we obtain the distance matrix with respect to the training and validation set, this gives the distance between each of the validation elements with each of the training samples
start = time.time()
dist_mat = DM(data_val,data_train)
end = time.time()
print("Preparing distance matrix took: ", (end-start)/60, " minutes")
sorted_dist_mat = np.zeros((data_train.shape[0],data_val.shape[0]))
start = time.time()
for val_element in range(dist_mat.shape[1]):
    sorted_dist_mat[:,val_element] = np.argsort(dist_mat[:,val_element],axis=0)
end = time.time()
print("Preparing arg sorted distance matrix took: ", (end-start)/60, " minutes")
# Using the K nearest neighbors to classify the data by using the actual class labels and the predicted class labels 
k = 5
sorted_dist_mat = sorted_dist_mat[0:k,:]
pred_mat  =  np.zeros((k,dist_mat.shape[1]))
for column in range(sorted_dist_mat.shape[1]):
    pred_mat[:,column] = labels_train[sorted_dist_mat[:,column].astype(int)][:,0]







# Concatenation of the labels with the data
data_train = np.concatenate((labels_train,data_train),axis=1)
data_val = np.concatenate((labels_val,data_val),axis=1)
data_test = np.concatenate((labels_test,data_test),axis=1)
# Before this a random shuffling of the training data can be done once I figure out why the np shuffle funcrtion is not working



