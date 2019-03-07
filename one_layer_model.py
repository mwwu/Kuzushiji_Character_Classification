import numpy as np
import tensorflow
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling1D
from keras import metrics
import keras
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
plt.hist(labels, bins=range(0, len(unique_labels)))
plt.show()
exit(0)

train_imgs = xtrain_imgs.reshape(xtrain_imgs.shape[0], 28, 28, 1).astype('float32')
test_imgs = xtest_imgs.reshape(xtest_imgs.shape[0], 28, 28, 1).astype('float32')


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

inputs = Input(shape=(28,28))
hidden = Conv2D(2, (3, 3), 2, padding="valid", activation='relu')(inputs)
hidden = GlobalAveragePooling1D()(hidden)
hidden = Dense(28, activation='sigmoid')(hidden)
prediction = Dense(10, activation='softmax')(hidden)

model = Model(inputs=inputs, outputs=prediction)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.1), metrics='accuracy')
model.summary()
exit(0)
history = model.fit(x=train_imgs, y=train_labels, epochs=10, batch_size=512, verbose=1, validation_split=.1)
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