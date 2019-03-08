import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras import metrics
from keras.layers.convolutional import Conv2D, AveragePooling2D
import keras
import matplotlib.pyplot as plt

# New imports
import datetime
from keras.callbacks import TensorBoard as TBCallback

# Layer imports
#from keras import backend as K
#from keras.layers import Layer
# Added class to identify single and double layer 
#from keras_layer import single_layer
#from keras_layer import double_layer

# Importing resnet.py made by raghakot
from resnet import ResnetBuilder

# custom resnet.py
from my_resnet import ResNet 


# UNCOMMENT BELOW FOR KMNIST
# KMNIST LOADING
# classes = 10
# with np.load("KMNIST/kmnist-test-imgs.npz") as data:
#         xtest_imgs = data['arr_0']
# with np.load("KMNIST/kmnist-test-labels.npz") as data:
#         test_labels = data['arr_0']
# print("Test imgs and labels loaded.")
# with np.load("KMNIST/kmnist-train-imgs.npz") as data:
#         xtrain_imgs = data['arr_0']
# with np.load("KMNIST/kmnist-train-labels.npz") as data:
#         train_labels = data['arr_0']
# print("Training imgs and labels loaded.")

# K49 LOADING
classes = 49
with np.load("K49/k49-test-imgs.npz") as data:
        xtest_imgs = data['arr_0']
with np.load("K49/k49-test-labels.npz") as data:
        test_labels = data['arr_0']
print("Test imgs and labels loaded.")
with np.load("K49/k49-train-imgs.npz") as data:
        xtrain_imgs = data['arr_0']
with np.load("K49/k49-train-labels.npz") as data:
        train_labels = data['arr_0']
print("Training imgs and labels loaded.")

# Data Preprocessing (unravel the image to a 1D vector)
# train_imgs = np.ndarray(shape=(len(xtrain_imgs), 784))
# test_imgs = np.ndarray(shape=(len(xtest_imgs), 784))
# for i in range(0, len(xtrain_imgs)):
#         train_imgs[i]=xtrain_imgs[i].ravel()
# for i in range(0, len(xtest_imgs)):
#         test_imgs[i]=xtest_imgs[i].ravel()

train_imgs = xtrain_imgs.reshape(xtrain_imgs.shape[0], 28, 28, 1).astype('float32')
test_imgs = xtest_imgs.reshape(xtest_imgs.shape[0], 28, 28, 1).astype('float32')
train_labels = keras.utils.to_categorical(train_labels, classes)
test_labels = keras.utils.to_categorical(test_labels, classes)


# All arrays are saved as ndarrays
# Input: 28x28 grayscale images each with value 0-255
# Output: Probability of image being one of 'classes' classes.
#  0   | Hidden Layer | 0
#  1   |              | 1
#  2   |              | 2
#  .   |              | .
#  .   |              | .
#  783 |              | 9
model = Sequential()

'''
#CONV LAYERS BEGIN
model.add(Conv2D(10, kernel_size=(3,5), padding="same", input_shape=(28,28,1), activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(10, kernel_size=(5,3), padding="same", activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

# 1/10 conv2d
model.add(Conv2D(3, kernel_size=(5,5), padding="same", activation='relu'))
model.add(Conv2D(7, kernel_size=(5,5), padding="same", activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

# 1/10 conv2d
model.add(Conv2D(9, kernel_size=(3,3), padding="same", activation='relu'))
model.add(Conv2D(12, kernel_size=(3,3), padding="same", activation='relu'))
model.add(Conv2D(40, kernel_size=(3,3), padding="same", activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
# CONV LAYERS END
model.add(Flatten())

#1/4 dense unit
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.1))
#model.add(Dense(units=200, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[metrics.categorical_accuracy])
'''

'''
Implement ResNet instead of Convolutional Model
ResnetBuilder function has everything.

basic_block for < 50 layers (thesis)
otherwise use bottleneck.

basick/bottleneck size defined at the end of resnet.py
'''
#previous github implementation
#model = ResnetBuilder.build_resnet_50((1, 28, 28), classes)

# new tutorial implementation
stages = [3,4,6]
filters = [64, 128, 256, 512]
model = ResNet.build(28, 28, 1, classes, stages, filters)

#original
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[metrics.categorical_accuracy])

#From GitHub
#model.compile(loss="categorical_crossentropy", optimizer="sgd")

model.summary()
# what I did to test epoch for last progress report
history = model.fit(x=train_imgs, y=train_labels, epochs=5, batch_size=16, verbose=1, validation_split=.1)

'''
Objective: Build a Keras Layer
'''
'''
# using custom layer from keras_layer
# Fix argument value. Maybe fix how layers are made in keras_layer.py
singleX = single_layer( Conv2D(10, kernel_size=(3,5), padding="same", input_shape=(28,28,1), activation = 'relu') )

# new. 
now = datetime.datetime.now()
logdir = f"logs/%d-%d-%d-%d" %(now.month, now.day, now.hour, now.minute)
#For test purpose, use 3 epochs only
history = model.fit(x=train_imgs, y=train_labels, epochs=3, batch_size=128, verbose=1, validation_split=.1,
        callbacks=[TBCallback(log_dir=logdir)])
#history = model.fit(x=train_imgs, y=train_labels, epochs=10, batch_size=128, verbose=1, validation_split=.1,
#        callbacks=[TBCallback(log_dir=logdir)])
'''

loss, accuracy = model.evaluate(test_imgs, test_labels)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
print("Test loss:", loss)
print("Test accuracy:", accuracy)
plt.show()
