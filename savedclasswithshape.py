import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
model_load=pickle.load(open("D:/classification/finalclassification1_saved","rb"))
model_load1=pickle.load(open("D:/classification/resnetname_saved","rb"))

dirs=os.listdir("D:/saibh/Pictures/CameraRoll")
imagesarr=[]
import os
ornament=['gold','silver']
ornamentshape = ['BRACELET', 'EARRINGS', 'NECKLACE', 'RINGS', 'WRISTWATCH'] 
for dir in dirs:
    imagesarr.append('D:saibh/Pictures/CameraRoll/'+dir )
    
x=len(imagesarr)-1   
i=imagesarr[x-5]
    
imgsize=180
image=tf.keras.utils.load_img(i,target_size=(imgsize,imgsize))
img_arr=tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)
predict=model_load.predict(img_bat)

predict1=model_load1.predict(img_bat)
score=tf.nn.softmax(predict)  
score1=tf.nn.softmax(predict1)
    
print("Length of ornament:", len(ornament))
print("Length of ornamentshape:", len(ornamentshape))    

  
print(i)

print("the ornament is {} {} with accuracy of {:0.2f} {:0.2f}".format(ornament[np.argmax(score)],ornamentshape[np.argmax(score1)],np.max(score)*100,np.max(score1)*100))
t=ornamentshape[np.argmax(score1)]


    


print("if the prediction is correct enter :1 \nif the prediction is wrong enter 0")
v=int(input("enter the value"))



if v==0:

    print("enter the index of correct class \n bangle:0 \n chain:1 \n earing:2 \ring:3")
    n=int(input("enter index no"))
    print("enter the index of correct class \n gold:0 \n silver:1")
    m=int(input("enter index no"))
    img_size = 180 
    

    img = cv2.imread(i)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  
    label=np.array([n])
    label1=np.array([m])
    model_load.compile(optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=["accuracy"])
    model_load.fit(img,label1,epochs=1) 
    model_load1.compile(optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    metrics=["accuracy"])
    model_load1.fit(img,label,epochs=1)                    
    
