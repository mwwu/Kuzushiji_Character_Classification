import numpy as np
import matplotlib.pyplot as plt

with np.load("KKanji/kkanji-imgs.npz") as data:
        imgs = data['arr_0']
with np.load("KKanji/kkanji-labels.npz") as data:
        labels = data['arr_0']
with np.load("KKanji/kkanji-unique-labels.npz") as data:
        unique_labels = data['arr_0']
print("imgs and labels loaded.")

classes = len(unique_labels)
def invert(x):
        return 1./x

hist = np.histogram(labels, bins=range(0, len(unique_labels)), density=True)
labels_count = np.zeros((classes, 1), dtype='int32')
def balance_select(array):
    for i in range(0, len(array)):
        labels_count[array[i]]+=1
balance_select(labels)
for i in range(0, classes):
    print(unique_labels[i], ": ", labels_count[i])
    # if labels_count[i]<100:
    #     print(unique_labels[i], "< 100")
    # elif labels_count[i]<200:
    #     print(unique_labels[i], "< 200")
    # elif labels_count[i]<300:
    #     print(unique_labels[i], "< 300")
    # elif labels_count[i]<400:
    #     print(unique_labels[i], "< 400")
    # elif labels_count[i]<500:
    #     print(unique_labels[i], "< 500")
    # elif labels_count[i]<600:
    #     print(unique_labels[i], "< 600")
    # elif labels_count[i]<700:
    #     print(unique_labels[i], "< 700")
    # elif labels_count[i]<800:
    #     print(unique_labels[i], "< 800")
    # elif labels_count[i]<900:
    #     print(unique_labels[i], "< 900")
    # elif labels_count[i]<1000:
    #     print(unique_labels[i], "< 1000")
