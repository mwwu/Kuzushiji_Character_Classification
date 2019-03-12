'''
Source:

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html?fbclid=IwAR2j_IwvDHoNgLyuY7jpr061Le9emGELlgFQO0302TyXG-AG6SIWERVOqTA

and 

https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720?fbclid=IwAR0Yhb2_FDpm97UzNSMfYD7_mDUQhNb4kWfDlMCJN1y9StPgziuNT3b9P1c

and

(currently used; need to fix some bugs)
https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720?fbclid=IwAR0Yhb2_FDpm97UzNSMfYD7_mDUQhNb4kWfDlMCJN1y9StPgziuNT3b9P1c
'''

'''
Didn't work. Use Megan's model that actually works
'''

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

train_generator = train_datagen.flow_from_directory(
        directory=r"./KMNIST/kmnist-train-imgs.npz/",
        target_size=(28,28),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical"#,
        #shuffle=True,
        #seed=42
)

'''
valid_generator = valid_datagen.flow_from_directory(
        directory=r"./KMNIST/kmnist-train-imgs.npz/",
        target_size=(28,28),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical"#,
        #shuffle=True,
        #seed=42
)
'''

test_generator = test_datagen.flow_from_directory(
        directory=r"./KMNIST/kmnist-test-imgs.npz/",
        target_size=(28,28),
        color_mode="grayscale",
        batch_size=1,
        class_mode="categorical"#,
        #shuffle=False,
        #seed=42
)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size 
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size 
model.fit_generator (generator=train_generator,
                     steps_per_epoch=STEP_SIZE_TRAIN,
                     validation_data=valid_generator,
                     validation_steps=STEP_SIZE_VALID,
                     epochs=10
)



model.evaluate_generator(generator=valid_generator)


# output prediction
test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv", index=False)


'''
# model from sample code
img_width, img_height = 28, 28

top_model_weight_path = ''
train_data_dir = './KKanji/'
validation_data_dir = ''
nb_train_samples = 300
nb_validation_samples = 100
epochs = 25
batch_size = 32



def bottleBeck():
    datagen = ImageDataGenerator(rescale=1. / 255)

    model = application.VGG
'''

'''
# model from intro code

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        color_mode="grayscale",
        class_mode="categorical",
        #rescale_1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

img = load_img('') # #kkanji thingy
imgArr = img_to_array(img) #numpy array? 3,150,150
x = x.reshape((1,) + x.shape) #numpy array? 1,3,150,150

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='', save_format='png'):
    i += 1
    
    if i > 20:
        break

'''
