import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras import metrics
from keras.callbacks import TensorBoard
import datetime
import keras
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

# Data Preprocessing
train_imgs = np.ndarray(shape=(len(xtrain_imgs), 784))
test_imgs = np.ndarray(shape=(len(xtest_imgs), 784))
for i in range(0, len(xtrain_imgs)):
        train_imgs[i]=xtrain_imgs[i].ravel()
for i in range(0, len(xtest_imgs)):
        test_imgs[i]=xtest_imgs[i].ravel()
classes = 10

train_labels = keras.utils.to_categorical(train_labels, classes)
test_labels = keras.utils.to_categorical(test_labels, classes)


# All arrays are saved as ndarrays
# Input: 28x28 grayscale images each with value 0-255
# Output: Probability of image being one of 10 classes.
#  0   | Hidden Layer | 0
#  1   |              | 1
#  2   |              | 2
#  .   |              | .
#  .   |              | .
#  783 |              | 9

inputs = Input(shape=(784,))
hidden = Dense(140, activation='relu')(inputs) #28*5
hidden = Dense(112, activation='relu')(hidden) #28*4
hidden = Dense(84, activation='relu')(hidden)  #28*3
hidden = Dense(56, activation='relu')(hidden)  #28*2
hidden = Dense(28, activation='relu')(hidden)  #28*1
prediction = Dense(10, activation='softmax')(hidden)

model = Model(inputs=inputs, outputs=prediction)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[metrics.categorical_accuracy])
model.summary()
dt = datetime.datetime.now()
tensorboard = TensorBoard(log_dir="logs/{}".format(dt.strftime("%b_%d_%Y_%I%M%p")), histogram_freq=1, batch_size=512, write_images=1)

history = model.fit(x=train_imgs, y=train_labels, epochs=30, batch_size=512, verbose=0, validation_split=.1, callbacks=[tensorboard])
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