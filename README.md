# Brain_Cancer_Detection
The Main Purpose Of This Project Is To Build a Robust CNN Model That Can Classify If The
Subject Has a Cancer Or Not Based On Brain MRI Scan Image With High Accuracy.

Model Accuracy-99.12%

# Tech Used:
HTML,CSS, Javascript, Bootstrap,Django,Python,SQLite, Algorithm- Convolution Neural Network(CNN)

# Open command prompt and enter the following Commands:
 1. pip install django
 2. pip install django-bootstrap3
 3. pip install tensorflow
 4. pip install pillow


# Open Command prompt, go to Project folder and enter the following command
 1. py manager.py runserver


# CNN Model Training Code
import warnings
warnings.filterwarnings('ignore')

!wget https://www.dropbox.com/s/6xmcksrq5g3ks16/BrainTumorDataSet.zip?dl=0

!unzip /content/BrainTumorDataSet.zip?dl=0

import numpy as np
import matplotlib.pyplot as plt 
import os
import math
import shutil
import glob

ROOT_DIR = "/content/Brain Tumor Data Set"
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
  number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))

number_of_images.items()

if not os.path.exists("./train"):
  os.mkdir("./train")

  for dir in os.listdir(ROOT_DIR):
    os.makedirs("./train/"+dir)
    for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)),
                                size=(math.floor(70/100*number_of_images[dir])-5),
                                replace=False):
      O = os.path.join(ROOT_DIR,dir,img)
      D = os.path.join('./train',dir)
      shutil.copy(O,D)
      os.remove(O)
else:
  print("Train Folder Exists")

if not os.path.exists("./val"):
  os.mkdir("./val")

  for dir in os.listdir(ROOT_DIR):
    os.makedirs("./val/"+dir)
    for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)),
                                size=(math.floor(15/100*number_of_images[dir])-5),
                                replace=False):
      O = os.path.join(ROOT_DIR,dir,img)
      D = os.path.join('./val',dir)
      shutil.copy(O,D)
      os.remove(O)
else:
  print("Val Folder Exists")

if not os.path.exists("./test"):
  os.mkdir("./test")

  for dir in os.listdir(ROOT_DIR):
    os.makedirs("./test/"+dir)
    for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)),
                                size=(math.floor(15/100*number_of_images[dir])-5),
                                replace=False):
      O = os.path.join(ROOT_DIR,dir,img)
      D = os.path.join('./test',dir)
      shutil.copy(O,D)
      os.remove(O)
else:
  print("Test Folder Exists")

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

def preprocessingImages(path):
  #image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, rescale=1/255, horizontal_flip=True)
  image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input, horizontal_flip=True)
  image = image_data.flow_from_directory(directory=path, target_size=(224,224), batch_size=32, class_mode='binary')

  return image

path = '/content/train'
train_data = preprocessingImages(path)

def preprocessingImages2(path):
  #image_data = ImageDataGenerator(rescale=1/255)
  image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
  image = image_data.flow_from_directory(directory=path, target_size=(224,224), batch_size=32, class_mode='binary')

  return image

path = '/content/test'
test_data = preprocessingImages2(path)

path = '/content/val'
val_data = preprocessingImages2(path)

import numpy
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense
from keras.models import Model, load_model
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet
import keras

base_model = MobileNet(input_shape=(224,224,3), include_top=False)

for layer in base_model.layers:
  layer.trainable = False

X = Flatten()(base_model.output)
X = Dense(units=1, activation="sigmoid")(X)

model = Model(base_model.input, X)

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor="val_accuracy", min_delta=0.02, patience=7, verbose=1, mode='auto')

mc = ModelCheckpoint(monitor="val_accuracy", filepath="trained_model.h5", verbose=1, save_best_only=True)

cb = [es,mc]

hist = model.fit_generator(train_data,
                           steps_per_epoch=20,
                           epochs=30,
                           validation_data=val_data,
                           validation_steps=16,
                           callbacks=cb)

model = load_model("/content/trained_model.h5")

accuracy = model.evaluate(test_data)[1]
print(f"Accuracy = {accuracy*100} %")

accuracy = model.evaluate(val_data)[1]
print(f"Accuracy = {accuracy*100} %")

accuracy = model.evaluate(train_data)[1]
print(f"Accuracy = {accuracy*100} %")
