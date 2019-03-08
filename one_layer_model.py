import numpy as np
import tensorflow
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, Flatten
from keras import metrics
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt

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

with np.load("KKanji/kkanji-imgs.npz") as data:
        imgs = data['arr_0']
with np.load("KKanji/kkanji-labels.npz") as data:
        labels = data['arr_0']
with np.load("KKanji/kkanji-unique-labels.npz") as data:
        unique_labels = data['arr_0']
print("imgs and labels loaded.")

hist = np.histogram(labels, bins=range(0, len(unique_labels)), density=True)
class_weights = hist[0]
# this is to get the weights of each class in the overall 3.8
imgs = imgs.reshape(imgs.shape[0], 64, 64, 1).astype('float32')
imgs_train, imgs_test, labels_train, labels_test = model_selection.train_test_split(imgs, labels, test_size=0.1)
classes = len(unique_labels)

labels_train = keras.utils.to_categorical(labels_train, num_classes=classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes=classes)


model = Sequential()

model.add(Conv2D(12, kernel_size=(3,3), padding="same", activation='relu', input_shape=(64,64,1)))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
model.summary()
history = model.fit(x=imgs_train, y=labels_train, epochs=10, batch_size=512, verbose=1, validation_split=.1, class_weight=class_weights)
loss, accuracy = model.evaluate(imgs_test, labels_test)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
print("Test loss:", loss)
print("Test accuracy:", accuracy)