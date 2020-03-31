import numpy as np
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

base_name = "/Users/megh/Work/misc/data/cifar-10-batches-py/data_batch_"
data = np.zeros((50000,3072))
labels = np.zeros((50000,1))
start = 0
end = 10000
for batch in range(1,6):
    data_dict_batch = unpickle(base_name+str(batch))
    data_batch = data_dict_batch[b'data']
    labels_batch = np.expand_dims(np.array(data_dict_batch[b'labels']),axis=1)
    data[start:end,:] = data_batch
    labels[start:end,:] = labels_batch
    start = start + 10000
    end = end + 10000
