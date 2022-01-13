# import important libraries-------

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras

xtrain=[]
ytrain=[]
xtest=[]

#storing images for training and testing-------

for i in glob.glob('C:\Users\US\Downloads\trantor_assignment\trantor\images\Unmarked_set'):
    img=cv2.imread(i) 
    color=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    color=cv2.resize(color,(128,128))
    xtrain.append(color)

for i in glob.glob('C:\Users\US\Downloads\trantor_assignment\trantor\images\marked_set'):
    img=cv2.imread(i) 
    b_w=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    b_w=cv2.resize(b_w,(128,128))    
    ytrain.append(b_w)

for i in glob.glob('C:\Users\US\Downloads\trantor_assignment\trantor\images\marked_set.jpg'):
    img=cv2.imread(i) 
    color=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    color=cv2.resize(color,(128,128))
    xtest.append(color)

#converting to numpy array
xtrain=np.array(xtrain)
ytrain=np.array(ytrain)
xtest=np.array(xtest)

#splitting for training---

xtrain,xval,ytrain,yval=train_test_split(xtrain,ytrain,test_size=0.33,random_state=42)

# Building Model (Unet++)---------

size=128
def unetpp(input_size = (size,size,1)):
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    
    

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)


    
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u31 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u31 = tf.keras.layers.concatenate([u31, c3])
    c31 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u31)
    c31 = tf.keras.layers.Dropout(0.2)(c31)
    c31 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c31)
    
    u21 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u21 = tf.keras.layers.concatenate([u21, c2])
    c21 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u21)
    c21 = tf.keras.layers.Dropout(0.2)(c21)
    c21 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c21)
    
    u22 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c31)
    u22 = tf.keras.layers.concatenate([u22, c2,c21])
    c22 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u22)
    c22 = tf.keras.layers.Dropout(0.2)(c22)
    c22 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c22)
    
    
    u11 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)
    u11 = tf.keras.layers.concatenate([u11, c1])
    c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = tf.keras.layers.Dropout(0.2)(c11)
    c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
    
    u12 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c21)
    u12 = tf.keras.layers.concatenate([u12, c1, c11])
    c12 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
    c12 = tf.keras.layers.Dropout(0.2)(c12)
    c12 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)
    
    u13 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c22)
    u13 = tf.keras.layers.concatenate([u13, c1, c11, c12])
    c13 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)
    c13 = tf.keras.layers.Dropout(0.2)(c13)
    c13 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
    
    

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3, c31])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2,c21,c22])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1,c11,c12,c13], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

model=unetpp((size,size,1)) # initialized---

# Train the model---------

best=tf.keras.callbacks.ModelCheckpoint('best.h5',monitor='val_loss')
model.compile(loss=['binary_crossentropy'],optimizer='adam',metrics=['mse'])
model.fit(xtrain,ytrain,validation_data=(xval,yval),batch_size=8,epochs=50,callbacks=[best])

# predict on test-----------

predictions=model.predict(xtest)

# final output----
for i in range(len(predictions)):
    temp=xtest[i].copy()
    temp=np.where((predictions[i]==255),(255,0,0),temp) # creating red box
    plt.imshow(temp)
    plt.show() # display detection on screen
