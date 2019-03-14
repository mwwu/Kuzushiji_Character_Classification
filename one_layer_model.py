import numpy as np
import tensorflow
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, Flatten
from keras import metrics
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt

with np.load("KMNIST/kmnist-test-imgs.npz") as data:
        xtest_imgs = data['arr_0']
with np.load("KMNIST/kmnist-test-labels.npz") as data:
        test_labels = data['arr_0']
print("Test imgs and labels loaded.")
with np.load("KMNIST/kmnist-train-imgs.npz") as data:
        xtrain_imgs = data['arr_0']
with np.load("KMNIST/kmnist-train-labels.npz") as data:
        train_labels = data['arr_0']
print("Training imgs and labels loaded.")
#
# with np.load("KKanji/kkanji-imgs.npz") as data:
#         imgs = data['arr_0']
# with np.load("KKanji/kkanji-labels.npz") as data:
#         labels = data['arr_0']
# with np.load("KKanji/kkanji-unique-labels.npz") as data:
#         unique_labels = data['arr_0']
# print("imgs and labels loaded.")

# hist = np.histogram(labels, bins=range(0, len(unique_labels)), density=True)
# class_weights = hist[0]
# # this is to get the weights of each class in the overall 3.8
# imgs = imgs.reshape(imgs.shape[0], 64, 64, 1).astype('float32')
# imgs_train, imgs_test, labels_train, labels_test = model_selection.train_test_split(imgs, labels, test_size=0.1)
# classes = len(unique_labels)
classes=49
train_imgs = xtrain_imgs.reshape(xtrain_imgs.shape[0], 28, 28, 1).astype('float32')
test_imgs = xtest_imgs.reshape(xtest_imgs.shape[0], 28, 28, 1).astype('float32')
train_labels = keras.utils.to_categorical(train_labels, classes)
test_labels = keras.utils.to_categorical(test_labels, classes)

model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['categorical_accuracy'])
model.summary()
history = model.fit(x=train_imgs, y=train_labels, epochs=15, batch_size=512, verbose=1, validation_split=.1)
loss, accuracy = model.evaluate(test_imgs, test_labels)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
print("Test loss:", loss)
print("Test accuracy:", accuracy)