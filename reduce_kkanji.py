import os
import imageio
import glob
import numpy as np

base_dir = "kkanji2"

train_imgs = []
train_labels = []
i = 0
for dirName, subdirList, fileList in os.walk(base_dir):
    arr = dirName.split("+")
    st = ""
    j = 0
    for im_path in glob.glob(dirName + "/*.png"):
        im = imageio.imread(im_path)
        #print(im.shape)

        train_imgs.append(im)
        st = arr[1]
        train_labels.append(st)
        j += 1
        if j > 100:
            break
    print(str(i) + ": " + st)
    i += 1
    if i > 100:
        break

unique_labels = list(set(train_labels))
train_labels[:] = [unique_labels.index(label) for label in train_labels]

np.savez("KKanji/kkanji-imgs", train_imgs)
print("saved training images")
np.savez("KKanji/kkanji-unique-labels", unique_labels)
print("saved unique labels")
np.savez("KKanji/kkanji-labels", train_labels)
print("saved training labels")
