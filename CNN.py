from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img = image.load_img("C:\\Users\\Aakash Dubey\\Desktop\\CNN\\basedata\\training\\correct_boundries\\01.JPG")
plt.imshow(img)

cv2.imread("C:\\Users\\Aakash Dubey\\Desktop\\CNN\\basedata\\training\\correct_boundries\\01.JPG").shape

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)
testing = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('C:\\Users\\Aakash Dubey\\Desktop\\CNN\\basedata\\training\\', target_size = (200,200), batch_size = 3, class_mode = 'binary')
 
validation_dataset = validation.flow_from_directory('C:\\Users\\Aakash Dubey\\Desktop\\CNN\\basedata\\validation\\', target_size = (200,200), batch_size = 3, class_mode = 'binary')

testing_dataset = testing.flow_from_directory('C:\\Users\\Aakash Dubey\\Desktop\\CNN\\basedata\\testing\\', target_size = (200,200), batch_size = 3, class_mode = 'binary')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape =(200,200,3)), tf.keras.layers.MaxPool2D(2,2), 
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'), tf.keras.layers.MaxPool2D(2,2), 
                                    #                                   
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'), tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation= 'relu'),
                                    ##
                                    tf.keras.layers.Dense(1, activation= 'sigmoid')])

model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics = ['accuracy'])
model_fit = model.fit(train_dataset, steps_per_epoch = 15, epochs = 50, validation_data = validation_dataset)

dir_path = ('C:\\Users\\Aakash Dubey\\Desktop\\CNN\\basedata\\testing\\')

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+' '+i, target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images =np.vstack([X])
    
    val = model.predict(images)
    if val==0:
        print('incorrect boundries')
        
    else:
        print('correct boundries')
        