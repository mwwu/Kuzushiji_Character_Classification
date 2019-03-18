import numpy as np
import tensorflow
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, Flatten
from keras import metrics
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf 
from tensorflow.python.lib.io import file_io
from keras.regularizers import l2

""" 
_____________________________________________

UNCOMMENT TO USE K49 DATASET
_____________________________________________

with np.load("K49/k49-imgs.npz") as data:
	imgs = data['arr_0']
with np.load("K49/k49-labels.npz") as data:
	labels = data['arr_0']
with np.load("K49/k49-unique-labels.npz") as data:
	unique_labels = data['arr_0']

"""

with np.load("KKanji/kkanji-imgs.npz") as data:
	imgs = data['arr_0']
with np.load("KKanji/kkanji-labels.npz") as data:
	labels = data['arr_0']
with np.load("KKanji/kkanji-unique-labels.npz") as data:
	unique_labels = data['arr_0']


""" 
_____________________________________________

UNCOMMENT TO USE KKANJI DATASET STORED ON GCP
_____________________________________________

f = BytesIO(file_io.read_file_to_string('gs://kuzushiji-classifier/kkanji-imgs.npz', binary_mode=True))
with np.load(f) as data:
    imgs = data['arr_0']

f = BytesIO(file_io.read_file_to_string('gs://kuzushiji-classifier/kkanji-labels.npz', binary_mode=True))
with np.load(f) as data:
    labels = data['arr_0']

f = BytesIO(file_io.read_file_to_string('gs://kuzushiji-classifier/kkanji-unique-labels.npz', binary_mode=True))
with np.load(f) as data:
    unique_labels = data['arr_0']
    
"""


print("imgs and labels loaded.")


hist = np.histogram(labels, bins=range(0, len(unique_labels)), density=True)
class_weights = hist[0]

imgs = imgs.reshape(imgs.shape[0], 64, 64, 1).astype('float32')
imgs_train, imgs_test, labels_train, labels_test = model_selection.train_test_split(imgs, labels, test_size=0.1)
classes = len(unique_labels)

labels_train = keras.utils.to_categorical(labels_train, num_classes=classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes=classes)

model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=classes, activation='softmax'))

""" 
_____________________________________________

UNCOMMENT TO TRY LARGER FEED FORWARD
_____________________________________________

model.add(Flatten(input_shape=(64,64,1)))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=classes, activation='softmax'))

"""

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['categorical_accuracy'])
model.summary()

history = model.fit(x=imgs_train, y=labels_train, epochs=15, batch_size=512, verbose=1, validation_split=.1)
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
