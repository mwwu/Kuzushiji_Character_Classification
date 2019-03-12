from final_model import imgs_train, imgs_test, labels_train, labels_test, classes, ResNet
from keras import Sequential, metrics, losses
model = Sequential()

# 34-layer UNIMPLEMENTED: NEED TO MAKE NON BOTTLENECK RESNET BLOCK
# stages = [2,2,2,2]
# filters = [64, 256, 512, 1024, 2048]
# 50-layer
stages = [3,4,6,3]
filters = [64, 256, 512, 1024, 2048]
# 101-layer
# stages = [3,8,23,3]
# filters = [64, 256, 512, 1024, 2048]
# 152-layer
# stages = [3,8,36,3]
# filters = [64, 256, 512, 1024, 2048]

model = ResNet.build(64, 64, 1, classes, stages, filters)

#original
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=[metrics.sparse_categorical_accuracy])

model.summary()

history = model.fit(x=imgs_train, y=labels_train, epochs=1, batch_size=256, verbose=1, validation_split=.1)

loss, accuracy = model.evaluate(imgs_test, labels_test)
print("loss:", loss)
print("accuracy:", accuracy)