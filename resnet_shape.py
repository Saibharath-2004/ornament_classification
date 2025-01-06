import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import Flatten, Dense
dirs1=os.listdir("D:/githubshapedataset/Jewellery-Classification/dataset/test")
basedir1=("D:/githubshapedataset/Jewellery-Classification/dataset/test")
batch=8
imgsize=180

train_ds1=tf.keras.utils.image_dataset_from_directory(
    basedir1,seed=123,validation_split=0.2,subset="training",label_mode="categorical",
    batch_size=batch,
    image_size=(imgsize,imgsize))
ornamentshape=train_ds1.class_names
ornamentshape



val_ds1=tf.keras.utils.image_dataset_from_directory(
    basedir1,seed=123,validation_split=0.2,subset="validation",
    batch_size=batch,label_mode="categorical",
    image_size=(imgsize,imgsize)
)
ornamentshape=train_ds1.class_names
ornamentshape
resnet_model=Sequential()
model1=tf.keras.applications.ResNet50(include_top=False,input_shape=(180,180,3),pooling="avg",classes=4,weights="imagenet")
for layer in model1.layers:
    layer.trainable=False
resnet_model.add(model1)
resnet_model.add(Flatten())
resnet_model.add(Dense(512,activation="relu"))
resnet_model.add(Dense(5,activation="softmax"))

resnet_model.summary()


resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
epochsize=10
history=resnet_model.fit(train_ds1,validation_data=val_ds1,epochs=epochsize)


imagesarr=[]
dirs=os.listdir("D:/saibh/Pictures/CameraRoll")
for dir in dirs:
    imagesarr.append('D:saibh/Pictures/CameraRoll/'+dir )
    
x=len(imagesarr)-1   
i=imagesarr[x-5]

image=tf.keras.utils.load_img("D:/githubshapedataset/Jewellery-Classification/dataset/training/WRISTWATCH/IMG-20181209-WA0035.jpg",target_size=(imgsize,imgsize))
img_arr=tf.keras.utils.img_to_array(image)
img_bat=tf.expand_dims(img_arr,0)
predict1=resnet_model.predict(img_bat)

    
           
        
    
score1=tf.nn.softmax(predict1)   
print(i)
print("the ornament shape is {} with accuracy of {:0.2f}".format(ornamentshape[np.argmax(score1)],np.max(score1)*100)) 

import pickle
pickle.dump(resnet_model,open("D:/classification/resnetname_saved","wb"))
