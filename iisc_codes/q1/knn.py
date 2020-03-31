import numpy as np
import pickle


something = 0
def get_k():
    return something

def L2_distance():
    
    return something

def cosine_distance():
    return something

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
base_name = "/Users/megh/Work/misc/data/cifar-10-batches-py/data_batch_"
for batch in range(1,5):
    data_dict = unpickle(base_name+str(batch))
    data = data_dict['data']
    labels = data_dict['labels']


