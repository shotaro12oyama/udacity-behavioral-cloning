import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split


#PARAMETERS
correction = 0.2    #collect angle for the images of left and right camera
CENTER = 0          #to use as camera_position in the image_and_angle_from_data function below 
LEFT = 1            #to use as camera_position in the image_and_angle_from_data function below
RIGHT = 2           #to use as camera_position in the image_and_angle_from_data function below




#read csv log data
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#split the log data for training and validation 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#
def image_and_angle_from_data(batch_sample, camera_position):
    name = './data/IMG/'+batch_sample[camera_position].split('/')[-1]
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, (3,3),0)                
    angle = float(batch_sample[3])

    return image, angle
        


#generator funtion for generating training and validating data
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image, center_angle = image_and_angle_from_data(batch_sample, CENTER)
                images.append(center_image)
                angles.append(center_angle)

                #using multiple cameras
                left_image, left_angle = image_and_angle_from_data(batch_sample, LEFT)
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)
                

                right_image, right_angle = image_and_angle_from_data(batch_sample, RIGHT)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)


            #involving flipping images
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)



#using NVIDIA model with adding Dropout
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,\
    nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')



### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


exit()
