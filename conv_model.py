import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras import metrics
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
import keras
from keras.callbacks import TensorBoard as TBCallback
import matplotlib.pyplot as plt
import datetime
import tensorboard
from final_model import imgs_train, imgs_test, labels_train, labels_test, classes, ResNet

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
# classes = 49
# with np.load("K49/k49-test-imgs.npz") as data:
#         xtest_imgs = data['arr_0']
# with np.load("K49/k49-test-labels.npz") as data:
#         test_labels = data['arr_0']
# print("Test imgs and labels loaded.")
# with np.load("K49/k49-train-imgs.npz") as data:
#         xtrain_imgs = data['arr_0']
# with np.load("K49/k49-train-labels.npz") as data:
#         train_labels = data['arr_0']
# print("Training imgs and labels loaded.")

# Data Preprocessing (unravel the image to a 1D vector)
# train_imgs = np.ndarray(shape=(len(xtrain_imgs), 784))
# test_imgs = np.ndarray(shape=(len(xtest_imgs), 784))
# for i in range(0, len(xtrain_imgs)):
#         train_imgs[i]=xtrain_imgs[i].ravel()
# for i in range(0, len(xtest_imgs)):
#         test_imgs[i]=xtest_imgs[i].ravel()
#
# train_imgs = xtrain_imgs.reshape(xtrain_imgs.shape[0], 28, 28, 1).astype('float32')
# test_imgs = xtest_imgs.reshape(xtest_imgs.shape[0], 28, 28, 1).astype('float32')
# train_labels = keras.utils.to_categorical(train_labels, classes)
# test_labels = keras.utils.to_categorical(test_labels, classes)


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
model.add(Conv2D(40, kernel_size=(5,5), padding="valid", input_shape=(64,64,1), activation = 'relu'))
model.add(Conv2D(40, kernel_size=(5,5), padding="valid", activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
#
model.add(Conv2D(50, kernel_size=(3,3), padding="same", activation='relu'))
model.add(Conv2D(60, kernel_size=(3,3), padding="same", activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(70, kernel_size=(5,5), padding="same", activation='relu'))
model.add(Conv2D(80, kernel_size=(3,3), padding="same", activation='relu'))
model.add(Conv2D(90, kernel_size=(1,1), padding="valid", activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(150, kernel_size=(3,3), padding="same", activation='relu'))
model.add(MaxPooling2D())
# CONV LAYERS END
model.add(Flatten())
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=[metrics.sparse_categorical_accuracy])
model.summary()
now = datetime.datetime.now()
# logdir = f"logs/%d-%d-%d-%d" %(now.month, now.day, now.hour, now.minute)
history = model.fit(x=imgs_train, y=labels_train, epochs=15, batch_size=128, verbose=1, validation_split=.1, class_weight=class_weights)
                    # , callbacks=[TBCallback(log_dir=logdir, update_freq=10000)])
model.save('test_model_viz.h5')
loss, accuracy = model.evaluate(imgs_test, labels_test)
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
print("Test loss:", loss)
print("Test accuracy:", accuracy)
plt.show()
