from final_model import imgs_train, imgs_test, labels_train, labels_test, classes, ResNet
from keras import Sequential, metrics
model = Sequential()

# new tutorial implementation
stages = [3,4,6]
filters = [64, 128, 256, 512]
model = ResNet.build(64, 64, 1, classes, stages, filters)

#original
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[metrics.categorical_accuracy])

#From GitHub
#model.compile(loss="categorical_crossentropy", optimizer="sgd")

model.summary()
# what I did to test epoch for last progress report
history = model.fit(x=imgs_train, y=labels_train, epochs=1, batch_size=256, verbose=1, validation_split=.1)
