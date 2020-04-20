import csv 
import os
import pandas as pd
#Creates a CSV file in the same folder where the experiment is being carried out
train_path = "/Users/megh/Work/github-repos/data/cifar-10-batches-py/cifar/train"
label_list = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
f1 = open('train.csv','w+')
patient_train_list = os.listdir(train_path)
f1.write('image, gt\n')
for patient in patient_train_list:
    f1 = open("train.csv", 'a')
    with f1:
        writer = csv.writer(f1)
        p = patient
        p = p.replace(".png","")
        p = p.replace("_","")
        p = ''.join([i for i in p if not i.isdigit()])
        if p=='airplane':
            label = 1
        if p=='autombile':
            label = 2
        if p=='bird':
            label = 3
        if p=='cat':
            label = 4
        if p=='deer':
            label = 5
        if p=='dog':
            label = 6
        if p=='frog':
            label = 7
        if p=='horse':
            label = 8
        if p=='ship':
            label = 9
        if p=='truck':
            label = 10
        writer.writerow([train_path + patient,label])