import numpy as np

with np.load("K49/k49-test-imgs.npz") as data:
    test_imgs = data['arr_0']
with np.load("K49/k49-test-labels.npz") as data:
    test_labels = data['arr_0']
with np.load("K49/k49-train-imgs.npz") as data:
    train_imgs = data['arr_0']
with np.load("K49/k49-train-labels.npz") as data:
    train_labels = data['arr_0']

print(test_imgs[0])
print(test_labels[0])


