from google.colab import drive
drive.mount('/content/gdrive')
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img ,img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image 
train_gen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,zoom_range=0.2,shear_range=0.2)
train_data = train_gen.flow_from_directory('/content/gdrive/MyDrive/Colab Notebooks/casting_data/train',class_mode='binary',batch_size=8,target_size=(64,64),color_mode='grayscale')
test_gen = ImageDataGenerator(rescale=1/255)
test_data = test_gen.flow_from_directory('/content/gdrive/MyDrive/Colab Notebooks/casting_data/test',class_mode='binary',batch_size=8,target_size=(64,64),color_mode='grayscale')
model = tf.keras.models.Sequential()
#convolution+pooling
model.add(tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same',input_shape=(64,64,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2))
#second layer
model.add(tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2))


model.add(tf.keras.layers.Flatten())  #flattening the image into 1d array




#creating nueral network 
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=128,activation='relu')) 
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))  
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(train_data,validation_data=test_data,epochs=10)
model.save('best_model.h5')
test_image = image.load_img('/content/gdrive/MyDrive/Colab Notebooks/custom_tester/3.jpeg',target_size=(64,64),color_mode='grayscale')
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
if result[0]<=0.5:
    print('Defective')
else :
    print('Not Defective')
import cv2
img = cv2.imread('/content/gdrive/MyDrive/Colab Notebooks/custom_tester/3.jpeg',0)
img = img/255 #rescaling
pred_img =img.copy()
plt.figure(figsize=(12,8))
plt.imshow(img,cmap='gray')
plt.show()
