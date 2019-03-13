
from keras.models import load_model

model = load_model("FirstModel.h5")

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def dataPrep(path):
    try:
        # Read GrayScale
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        plt.imshow(img,cmap = "gray")
        plt.show()
        # Resize
        img = cv2.resize(img,(50,50))
        # Change to npArray
        X = np.array(img).reshape(-1,50,50,1)
        # Normalization
        X = X/255.0
        return X
    except Exception as e:
        print(str(e))

def predict(model,X):
    if(model.predict(X)<0.5):
        return "cat"
    else:
        return "dog"


predict(model,dataPrep("images.jpg"))


