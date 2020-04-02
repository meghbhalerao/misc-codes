import numpy as np
import pickle
import time 

def DM(data_val,data_train):
    dist_mat = np.zeros((data_train.shape[0],data_val.shape[0]))
    for sample_val in range(data_val.shape[0]):
        for sample_train in range(data_train.shape[0]):
            dist_mat[sample_train,sample_val] = L2_distance(data_val[sample_val,1:], data_train[sample_train,1:])         
    return dist_mat

def L2_distance(mat1,mat2):
    dist = np.linalg.norm(mat1-mat2)    
    return dist

def get_most_frequent(array):
    counts = np.bincount(array)
    return np.argmax(counts)

something = 0
def cosine_distance(mat1,mat2):
    mat1 = mat1.flatten()
    mat2 = mat2.flatten()
    dist = (mat1*mat2)/(np.linalg.norm(mat1)*np.linalg.norm(mat2))
    return dist

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
dist_mat_val = DM(data_val,data_train)
end = time.time()
print("Preparing distance matrix took: ", (end-start)/60, " minutes")
sorted_dist_mat_val = np.zeros((data_train.shape[0],data_val.shape[0]))
start = time.time()
for val_element in range(dist_mat_val.shape[1]):
    sorted_dist_mat_val[:,val_element] = np.argsort(dist_mat_val[:,val_element],axis=0)
end = time.time()
print("Preparing arg sorted distance matrix took: ", (end-start)/60, " minutes")
# Using the K nearest neighbors to classify the data by using the actual class labels and the predicted class labels 
k = 5
sorted_dist_mat_val = sorted_dist_mat_val[0:k,:]
pred_mat_val  =  np.zeros((k,dist_mat_val.shape[1]))
for column in range(sorted_dist_mat_val.shape[1]):
    pred_mat_val[:,column] = labels_train[sorted_dist_mat_val[:,column].astype(int)][:,0]
    

pred_val = []
for column in range(pred_mat_val.shape[1]):
    pred_val.append(get_most_frequent(pred_mat_val[:,column].astype(int)))

pred_val =  np.array([pred_val]).T
acc_vec_val =  (pred_val - labels_val_val).astype(int)
acc_val = (pred_val.shape[0] - np.count_nonzero(acc_vec_val))/pred_val.shape[0]
print("Accuracy on validation set of CIFAR-10 is: ",acc_val*100," percent")

# Calculating the accuracy on the testing data
start = time.time()
dist_mat_test = DM(data_test,data_train)
end = time.time()
print("Preparing distance matrix took: ", (end-start)/60, " minutes")
sorted_dist_mat_test = np.zeros((data_train.shape[0],data_test.shape[0]))
start = time.time()
for test_element in range(dist_mat_test.shape[1]):
    sorted_dist_mat_test[:,test_element] = np.argsort(dist_mat_test[:,test_element],axis=0)
end = time.time()
print("Preparing arg sorted distance matrix took: ", (end-start)/60, " minutes")
# Using the K nearest neighbors to classify the data by using the actual class labels and the predicted class labels 
k = 5
sorted_dist_mat_test = sorted_dist_mat_test[0:k,:]
pred_mat_test  =  np.zeros((k,dist_mat_test.shape[1]))
for column in range(sorted_dist_mat_test.shape[1]):
    pred_mat_test[:,column] = labels_train[sorted_dist_mat_test[:,column].astype(int)][:,0]
    

pred_test = []
for column in range(pred_mat_test.shape[1]):
    pred_test.append(get_most_frequent(pred_mat_test[:,column].astype(int)))

pred_test =  np.array([pred_test]).T
acc_vec_test =  (pred_test - labels_test).astype(int)
acc_test = (pred_test.shape[0] - np.count_nonzero(acc_vec_test))/pred_test.shape[0]
print("Accuracy on test set of CIFAR-10 is: ",acc_test*100," percent")
# Finding the best K value using brute force 
search_range =  [3,5,7,9,11,13]
acc_list = []
for k in search_range:
    k = int(k)
    sorted_dist_mat_val = sorted_dist_mat_val[0:k,:]
    pred_mat_val  =  np.zeros((k,dist_mat_val.shape[1]))
    for column in range(sorted_dist_mat_val.shape[1]):
        pred_mat_val[:,column] = labels_train[sorted_dist_mat_val[:,column].astype(int)][:,0]   

    pred_val = []
    for column in range(pred_mat_val.shape[1]):
        pred_val.append(get_most_frequent(pred_mat_val[:,column].astype(int)))

    pred_val =  np.array([pred_val]).T
    acc_vec_val =  (pred_val - labels_val_val).astype(int)
    acc_val = (pred_val.shape[0] - np.count_nonzero(acc_vec_val))/pred_val.shape[0]
    print("Accuracy on validation set of CIFAR-10 is: ",acc_val*100," percent")
    acc_list.append(acc_val)
acc_list = np.array(acc_list)
acc_list = np.argsort(-1*acc_list)
print("The best validation accuracy occurs at k = ", 2*acc_list[0] + 1) 
    






























