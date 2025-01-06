import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
dirs=os.listdir("D:/classification/images")


basedir="D:/classification/images"
imgsize=180
batch=8
train_ds=tf.keras.utils.image_dataset_from_directory(
    basedir,seed=123,validation_split=0.2,subset="training",
    batch_size=batch,
    image_size=(imgsize,imgsize)
)
val_ds=tf.keras.utils.image_dataset_from_directory(
    basedir,seed=123,validation_split=0.2,subset="validation",
    batch_size=batch,
    image_size=(imgsize,imgsize)
)
ornament=train_ds.class_names
ornament


for index, class_name in enumerate(vehicle):
    print(f"Class: {class_name}, Index: {index}")
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
    for i in range(7):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(ornament[labels[i]])
        plt.axis("off")

model=Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding="same",activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding="same",activation="relu"),
    layers.MaxPooling2D() ,
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(2)])
imagesarr=[]
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
epochsize=35

history=model.fit(train_ds,validation_data=val_ds,epochs=epochsize)


import os
dirs=os.listdir("D:/saibh/Pictures/CameraRoll")

for dir in dirs:
    imagesarr.append('D:saibh/Pictures/CameraRoll/'+dir )
    
x=len(imagesarr)-1
i=imagesarr[x-5]
    

image=tf.keras.utils.load_img(i,target_size=(imgsize,imgsize))
img_arr=tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)
predict=model.predict(img_bat)
    
score=tf.nn.softmax(predict)    
           
        
    
print(i)
print("the ornament is {} with accuracy of {:0.2f}".format(ornament[np.argmax(score)],np.max(score)*100))  



import pickle
pickle.dump(model,open("D:/classification/finalclassification1_saved","wb"))
