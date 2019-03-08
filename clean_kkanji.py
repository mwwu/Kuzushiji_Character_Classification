import os
import imageio
import glob
import numpy as np

base_dir = "KKanji/kkanji2"

train_imgs = []
train_labels = []
for dirName, subdirList, fileList in os.walk(base_dir):
    for im_path in glob.glob(dirName + "/*.png"):
        im = imageio.imread(im_path)
        # print(im.shape)

        train_imgs.append(im)
        train_labels.append(dirName[-4:])
    print(dirName)


unique_labels = list(set(train_labels))
train_labels[:] = [unique_labels.index(label) for label in train_labels]

np.savez("KKanji/kkanji-imgs", train_imgs)
np.savez("KKanji/kkanji-unique-labels", unique_labels)

np.savez("KKanji/kkanji-labels", train_labels)
