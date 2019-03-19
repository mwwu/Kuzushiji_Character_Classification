from final_model import imgs_train, imgs_test, labels_train, labels_test, classes, ResNet
from keras import Sequential, metrics, losses
model = Sequential()

# 50 layer resnet
stages = [3,4,6,3]
filters = [64, 256, 512, 1024, 2048]


"""
_____________________________________________

FUTURE WORK: MORE RESNET IDEAS  (NEED TO MAKE BOTTLENECK FIRST)
_____________________________________________
# 34-layer 
stages = [2,2,2,2]
filters = [64, 256, 512, 1024, 2048]

# 101-layer
stages = [3,8,23,3]
filters = [64, 256, 512, 1024, 2048]

# 152-layer
stages = [3,8,36,3]
filters = [64, 256, 512, 1024, 2048]

"""





"""
_____________________________________________

UNCOMMENT BELOW TO USE K49 DATASET
_____________________________________________
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
train_imgs = np.ndarray(shape=(len(xtrain_imgs), 784))
test_imgs = np.ndarray(shape=(len(xtest_imgs), 784))
for i in range(0, len(xtrain_imgs)):
	train_imgs[i]=xtrain_imgs[i].ravel()
for i in range(0, len(xtest_imgs)):
	test_imgs[i]=xtest_imgs[i].ravel()

train_imgs = xtrain_imgs.reshape(xtrain_imgs.shape[0], 28, 28, 1).astype('float32')
test_imgs = xtest_imgs.reshape(xtest_imgs.shape[0], 28, 28, 1).astype('float32')
train_labels = keras.utils.to_categorical(train_labels, classes)
test_labels = keras.utils.to_categorical(test_labels, classes)

model = ResNet.build(28, 28, 1, classes, stages, filters)


"""

model = ResNet.build(64, 64, 1, classes, stages, filters)


#original
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=[metrics.sparse_categorical_accuracy])

model.summary()

history = model.fit(x=imgs_train, y=labels_train, epochs=1, batch_size=256, verbose=1, validation_split=.1)

loss, accuracy = model.evaluate(imgs_test, labels_test)
print("loss:", loss)
print("accuracy:", accuracy)