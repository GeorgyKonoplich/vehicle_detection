import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import image as imageproc
import pandas as pd

#path_to_raw_data = "/home/konoplich/workspace/projects/BloodTranscriptome/scripts/data/vehicle_detection/data/raw/" #ubuntu
path_to_raw_data = "C:/workspace/ml/vehicle_detection/data/raw/" #windows

true_images = imageproc.list_pictures(path_to_raw_data + "data_true/")
list_true_images = [imageproc.img_to_array(imageproc.load_img(x)) for x in true_images]
false_images = imageproc.list_pictures(path_to_raw_data + "data_false/")
list_false_images = [imageproc.img_to_array(imageproc.load_img(x)) for x in false_images]

# this will do preprocessing and realtime data augmentation
datagen = imageproc.ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


true_images = np.array(list_true_images)
datagen.fit(true_images)
for j in range(100):
    images_transform = [datagen.standardize(datagen.random_transform(x)) for x in true_images]
    list_true_images += images_transform

false_images = np.array(list_false_images)
datagen.fit(false_images)
for j in range(100):
    images_transform = [datagen.standardize(datagen.random_transform(x)) for x in false_images]
    list_false_images += images_transform



true_labels = [1.] * len(list_true_images)
false_labels = [0.] * len(list_false_images)
images = list_true_images + list_false_images
labels = true_labels + false_labels
data = [(images[i], labels[i]) for i in range(len(images))]

data_train = np.array(images)
target_train = np.array(labels)

np.save("C:/workspace/ml/vehicle_detection/data/processed/train_data_new", data_train)
np.save("C:/workspace/ml/vehicle_detection/data/processed/train_target_new", target_train)