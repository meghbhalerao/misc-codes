data_test = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/data_test.txt")
labels_test = np.loadtxt("/Users/megh/Work/misc/data/cifar-10-batches-py/txt_data/labels_test.txt")
labels_test = np.expand_dims(labels_test,axis=1)
data_test = np.concatenate((labels_test,data_test))
