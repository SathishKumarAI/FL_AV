'''
Dataset Total Size: 1326
Training: 929
Testing: 132
Valid: 265
Percentage Division: (70,20,10)(train, valid, test)

#data size is 1143images of weapons(OLD)
#  800 for training and 228 for validation and 115 for test

Image size : 3165
Train dataset : 2532
Test dataset : 625
'''
import os
#from os import listdir
#from os.path import isfile, join
import numpy as np
directory = r'C:\Users\SIU856522160\Downloads\yolov5\data\val'

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
count_train = 1
count_test=1
count_valid=1
train = open(r"C:\Users\SIU856522160\Downloads\final repo\yolov5_10k\dataset\train.txt", "w+")
#test = open("test.txt", "w+")
val = open(r"C:\Users\SIU856522160\Downloads\final repo\yolov5_10k\dataset\valid.txt", "w+")
# list all files in dir
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
print(files)
files_taken=[]
# select 0.7 of the files randomly for train set and 0.2 valid set and 0.1 for test set
random_files_train = np.random.choice(files, int(len(files)*.8))
#print(random_files)
for filename in random_files_train:
    if ( filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG") ) and count_train <=1280:
        train.write("./data/val/"+filename+"\n")
        count_train += 1
rest=Diff(files, random_files_train)

random_files_valid = np.random.choice(rest, int(len(files)*.8))
    #print(filename)
for filename in random_files_valid:
    if ( filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG") ) and count_valid <=320:
        val.write("./data/val/"+filename+"\n")
        count_valid += 1

'''
test_dataset=Diff(rest, random_files_valid)

for filename in test_dataset:
    if ( filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG") ) and count_test <= 101:
        test.write("./dataset/"+filename+"\n")
        count_test += 1
'''
train.close()
#test.close()
val.close()
