import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

DATADIR = "./PetImages"
CATEGORIES = ["Cat","Dog"]

IMGSIZE = 50
trainingData = []

# Read Data in Grayscale From cv2

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            # RESIZE IMG
            img_array = cv2.resize(img_array, (IMGSIZE,IMGSIZE))
            # Label
            label = CATEGORIES.index(category)
            #training data
            trainingData.append([img_array,label])
        except Exception as e:
            pass


# Shuffle Data

import random
random.shuffle(trainingData)

# split train and test set

X = []
y = []
for train in trainingData:
    X.append(train[0])
    y.append(train[1])


# Convert list X to np array ( keras cannot feed in with python list )

X = np.array(X).reshape(-1,IMGSIZE,IMGSIZE,1) # (number of sample, imgsize, imgsize, 1 channel because gray scale)


# save preprocessed data

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out) # (list, fileName)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out) # (list, fileName)
pickle_out.close()



