import numpy as np
import pickle
import time 

something = 0
def get_k():
    return something

def L2_distance():
    
    return something

def cosine_distance():
    return something

start = time.time()
data_train = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/data_train.txt")
labels_train = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/labels_train.txt")
data_test = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/data_test.txt")
labels_test = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/labels_test.txt")
end = time.time()
print("Loading data took: ", (end-start)/60, " minutes")
# Expanding dimenrtions of labels for concatenation with image data
labels_train = np.expand_dims(labels_train,axis=1)
labels_test = np.expand_dims(labels_test,axis=1)
