import os
import imageio
import glob
#base_dir = "\kkanji2\U+4C61\e6eeb23552e1a21c.png"
base_dir = "kkanji2"

for dirName, subdirList, fileList in os.walk(base_dir):
    # print(dirName[10:]) # labels
    for im_path in glob.glob(dirName + "\\*.png"):
        # print(im_path)
        im = imageio.imread(im_path)
        print(im.shape)
        # TODO: take im and make it into a numpy with dirName[10:] as the label name.