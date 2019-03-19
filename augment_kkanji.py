import numpy as np
from keras.models import Sequential
from keras import metrics
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import imageio

with np.load("KKanji/kkanji-imgs.npz") as data:
        imgs = data['arr_0']
with np.load("KKanji/kkanji-labels.npz") as data:
        labels = data['arr_0']
with np.load("KKanji/kkanji-unique-labels.npz") as data:
        unique_labels = data['arr_0']
print("imgs and labels loaded.")

classes = len(unique_labels)

labels_count = np.zeros((classes, 1), dtype='int32')
def balance_select(array):
    for i in range(0, len(array)):
        labels_count[array[i]]+=1
balance_select(labels)

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.3,
        zoom_range=0.15,
        fill_mode='nearest')

ideal_num_classes = 20

for item in range(0, classes):
    if (labels_count[item] > 0 and ideal_num_classes > labels_count[item]):
        i = 0
        for filename in os.listdir('kkanji2/U+' + unique_labels[item]):
            if filename.endswith('.png'):
                numNewImgs = int((ideal_num_classes-labels_count[item])/labels_count[item])

                im_path = 'kkanji2/U+' + unique_labels[item] + "/"+ filename
                x = imageio.imread(im_path)
                x = x.reshape((1,64,64,1))

                j = 0
                for batch in datagen.flow(x,
                    save_to_dir='/Users/megan/projects/kuzushiji_model_training/kkanji2/U+' + unique_labels[item],
                    save_prefix="gen_" + filename, save_format='png'):
                    j += 1
                    i += 1
                    if j > numNewImgs:
                        break
                    if i > ideal_num_classes:
                        break
                if i > ideal_num_classes:
                    break
        print(str(item) + ":" + unique_labels[item])
