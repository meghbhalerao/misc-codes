import numpy as np
import pandas as pd 
df = pd.read_csv("names.txt",header=None)
l = df[0].values.tolist()
def closestBinarySearch(l,name,left,right):
    while(right>left):
        mid = int((left+right)/2)
        if name > str(l[mid]).lower():
            left = mid + 1
        if name < str(l[mid]).lower():
            right = mid - 1
        if name == str(l[mid]).lower():
            return mid
    return mid

name = "megh"

cl = closestBinarySearch(l,name,0,len(l))
print("The closest words are, first word is the closest word and the rest are in no particular order: ", l[cl],l[cl-1],l[cl-2],l[cl+1],l[cl+2])
    
    