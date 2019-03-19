# Kuzushiji_Model_Training
Dependencies:
- NumPy
- Keras/Tensorflow backend
- Pandas
- scikit-learn
- matplotlib
- requests


# K49 | Feed Forward
1. vi model_feed_forward.py
	- uncomment 49 sections, comment kkanji section (see comments in the code)
	- change image shape for 28x28 (see comments in the code)
2. python3 model_feed_forward.py

# K49 | CNN
1. vi model_feed_forward.py
	- uncomment 49 sections, comment kkanji section (see comments in the code)
2. python3 model_cnn.py

# K49 | ResNet
1. vi model_resnet.py
	- uncomment 49 sections, comment kkanji section (see comments in the code)
2. python3 model_resnet.py



# KKanji | Data Preprocessing

We included the processed .npz files of the datasets in their respective folders, so you can skip this step. Otherwise, you run "python3 download_data.py" to get all of the images for KKanji.

1. python3 clean_kkanji.py (this extracts the file names, generates labels for each image, and create a .npz from the pngs)
2. python3 augment_kkanji.py (oversampling with data augmentation)
3. python3 clean_kkanji.py  (this extracts the file names, generates labels for each image, and create a .npz from the pngs)
4. python3 reduce_kkanji.py (undersampling with data augmentation)


# KKanji | Feed Forward
1. vi model_feed_forward.py
	- comment out k49 sections (see comments in the code)
	- change image shape for 64x64 (see comments in the code)
2. python3 model_feed_forward.py

# KKanji | CNN
1. vi model_feed_forward.py
	- comment out k49 sections (see comments in the code)
2. python3 model_cnn.py

# KKanji | ResNet
1. vi model_resnet.py
	- comment out k49 sections (see comments in the code)
2. python3 model_resnet.py
	- we included the output in model_output.txt, because this model takes a few days to train

