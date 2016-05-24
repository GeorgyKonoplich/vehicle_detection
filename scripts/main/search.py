import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import image as imageproc
import pandas as pd
import sys
import pickle
from keras.models import model_from_json
from keras.optimizers import SGD
from PIL import Image

#path_to_project = "/home/konoplich/workspace/projects/BloodTranscriptome/scripts/data/vehicle_detection/" #ubuntu
path_to_project = "C:/workspace/ml/graduate_work/vehicle_detection/" #windows"

path_to_model = path_to_project + "/models/dnn_new"

path_to_photo = path_to_project + "scripts/main/photo1.jpg"#sys.argv[1]

path2 = path_to_project + "scripts/main/99.jpg"#sys.argv[1]

def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pickle.load(open(file_from, 'rb'))
    nn = model_from_json(nn_arch)
    nn.set_weights(nn_weights_path)
    return nn

def get_image(x):
	image = imageproc.array_to_img(x)
	return image

def get_array(image):
	x = imageproc.img_to_array(image)
	return x

def load_image(path):
	image = imageproc.load_img(path, grayscale=True)
	return image

def save_photos(imgs, path):
	cnt = 0
	for x in imgs:
		cnt += 1
		image = get_image(x)
		image.save(path + str(cnt) + '.jpg')	

def resize_imagearrays(imgs, size):
	img_res = []
	for x in imgs:
		image = get_image(x)
		image.thumbnail(size, Image.ANTIALIAS)
		x = imageproc.img_to_array(image)
		img_res.append(x)
	img_res = np.array(img_res)
	return img_res

def get_windows(image, size = [48, 48], step = 1, factor = 1.4, min_din = 1.5):
	x = 0
	y = 0
	imgs = []
	img_array = get_array(image)
	while x + size[0] <= image.size[1]:
		y = 0
		while y + size[1] <= image.size[0]:
			l = x + size[0]
			r = y + size[1]
			imgs.append(img_array[0:1, x:l, y:r])
			y += step
		x += step
	imgs = np.array(imgs)
	return imgs

image = load_image(path_to_photo)
img_array = get_array(image)


model = load_neural_network(path_to_model)

sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


imgs = get_windows(image, size = [100, 100], step = 50)

save_photos(imgs, 'res1/')
imgs = resize_imagearrays(imgs, (48, 48))
save_photos(imgs, 'res2/')

arr  = model.predict(imgs)

mx = 1.0

cnt = 0
for x in arr:
	cnt += 1
	if mx == x:
		get_image(imgs[cnt - 1]).save("ress/" +str(cnt) + ".jpg")

'''
#x = imageproc.array_to_img(arr).save('aug_11.jpg')
img1 = imageproc.img_to_array(imageproc.load_img(path2, grayscale=True))

ar2 = []
arr4 = np.zeros([1, 48, 48])

ar2.append(img1)
ar2.append(arr4)

ar3 = np.array(ar2)
print(ar3.shape)
att = model.predict(ar3)
for x in att:
	print("%.4f" % x)
'''