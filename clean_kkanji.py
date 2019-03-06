import os
import imageio
import glob
import string
import unicodedata
import fnmatch
import numpy as np

#base_dir = "\kkanji2\U+4C61\e6eeb23552e1a21c.png"

base_dir = "minikanji"

train_imgs = []
train_labels = []
for dirName, subdirList, fileList in os.walk(base_dir):
    for im_path in glob.glob(dirName + "/*.png"):
        #print(im_path)

        im = imageio.imread(im_path)
        #print(im.shape)

        train_imgs.append(im)
        train_labels.append(dirName[-4:])


unique_labels = list(set(train_labels))
train_labels[:] = [unique_labels.index(label) for label in train_labels]

np.savez("KKanji/kkanji-imgs", train_imgs)


np.savez("KKanji/kkanji-labels", train_labels)

with np.load("kkanji/kkanji-imgs.npz") as data:
    for i in data['arr_0']:
        print(i.shape)

with np.load("kkanji/kkanji-labels.npz") as data:
    for i in data['arr_0']:
        print(i)

              # .decode(encoding='unicode-escape'))
              # .decode(encoding='utf-16'))
        # TODO: take im and make it into a numpy with dirName[10:] as the label name.

        #get folders, get file names, TODO: image processing