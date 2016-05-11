import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import image as imageproc
import pandas as pd
import sys
import pickle
from keras.models import model_from_json
from keras.optimizers import SGD

#path_to_project = "/home/konoplich/workspace/projects/BloodTranscriptome/scripts/data/vehicle_detection/" #ubuntu
path_to_project = "C:/workspace/ml/vehicle_detection/" #windows"

path_to_model = path_to_project + "/models/dnnnew"

#if (len(sys.argv) == 3):
path_to_photo = path_to_project + "scripts/main/photo.jpg"#sys.argv[1]

path2 = path_to_project + "scripts/main/image3.jpg"#sys.argv[1]


print(path_to_photo)

image = imageproc.img_to_array(imageproc.load_img(path_to_photo, grayscale=True))

print(image.shape)
print(type(image))

#img = imageproc.array_to_img(image)
#img.save("out.jpg")
def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pickle.load(open(file_from, 'rb'))
    nn = model_from_json(nn_arch)
    nn.set_weights(nn_weights_path)
    return nn


model = load_neural_network(path_to_model)

sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(image.shape)

arr = image[0:1, 0:48, 0:48]
#x = imageproc.array_to_img(arr).save('aug_11.jpg')
img1 = imageproc.img_to_array(imageproc.load_img(path2, grayscale=True))

ar2 = []
arr4 = np.zeros([1, 48, 48])
ar2.append(arr)
ar2.append(img1)
ar2.append(arr4)

ar3 = np.array(ar2)
print(ar3.shape)

print(model.predict(ar3))