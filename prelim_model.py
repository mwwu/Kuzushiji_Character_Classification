import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras import metrics
from keras.layers.convolutional import Conv2D, AveragePooling2D
import keras
import matplotlib.pyplot as plt

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

#CONV LAYERS BEGIN
model.add(Conv2D(4, kernel_size=(7,7), padding="valid", input_shape=(28,28,1), activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(4, kernel_size=(7,7), padding="same", activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(10, kernel_size=(5,5), padding="same", activation='relu'))
model.add(Conv2D(20, kernel_size=(5,5), padding="same", activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(40, kernel_size=(3,3), padding="same", activation='relu'))
model.add(Conv2D(70, kernel_size=(1,1), padding="same", activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
# CONV LAYERS END
model.add(Flatten())

model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[metrics.categorical_accuracy])
model.summary()
history = model.fit(x=train_imgs, y=train_labels, epochs=5, batch_size=64, verbose=1, validation_split=.1)
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
