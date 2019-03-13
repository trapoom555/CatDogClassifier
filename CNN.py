import pickle
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten
import time
from keras.callbacks import TensorBoard

NAME = "CatDogCNN-{}".format(int(time.time())) # name of log for tensor board
tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME)) # create TensorBoard

# Load prepared data

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

# Normalization Data

X = X/255.0

# modeling

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("sigmoid"))

model.add(Dense(1))
model.add(Activation("sigmoid"))


model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])


# Can view on tensor board

model.fit(X,y,batch_size = 32,validation_split = 0.1, epochs = 10, callbacks = [tensorboard])

# save model

model.save('FirstModel.h5')




