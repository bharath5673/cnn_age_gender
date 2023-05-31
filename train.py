import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

path = "/content/UTKFace/UTKFace"
pixels = []
age = []
gender = []
for img in os.listdir(path):
  ages = img.split("_")[0]
  genders = img.split("_")[1]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  pixels.append(np.array(img))
  age.append(np.array(ages))
  gender.append(np.array(genders))
age = np.array(age,dtype=np.int64)
pixels = np.array(pixels)
gender = np.array(gender,np.uint64)

x_train,x_test,y_train,y_test = train_test_split(pixels,age,random_state=100)
x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(pixels,gender,random_state=100)


input = Input(shape=(200, 200, 3))
conv1 = Conv2D(64, (3, 3), activation="relu")(input)
pool1 = MaxPool2D((2, 2))(conv1)
conv2 = Conv2D(32, (3, 3), activation="relu")(pool1)
pool2 = MaxPool2D((2, 2))(conv2)
conv3 = Conv2D(16, (3, 3), activation="relu")(pool2)
pool3 = MaxPool2D((2, 2))(conv3)
flatten = Flatten()(pool3)

# Age prediction
age_l = Dense(32, activation="relu")(flatten)
age_l = Dense(16, activation="relu")(age_l)
age_l = Dense(1)(age_l)

# Gender prediction
gender_l = Dense(32, activation="relu")(flatten)
gender_l = Dense(16, activation="relu")(gender_l)
gender_l = Dropout(0.5)(gender_l)
gender_l = Dense(2, activation="softmax")(gender_l)

model = Model(inputs=input, outputs=[age_l, gender_l])
model.compile(optimizer="adam", loss=["mse", "sparse_categorical_crossentropy"], metrics=['mae', 'accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=5)

save = model.fit(x_train, [y_train, y_train_2], validation_data=(x_test, [y_test, y_test_2]),
                #  epochs=50, callbacks=[checkpoint, early_stop])
                 epochs=50, callbacks=[checkpoint])


model.save("final_model.h5")
