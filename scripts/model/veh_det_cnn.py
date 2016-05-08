import numpy as np
np.random.seed(1337)
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import pandas as pd
import sklearn.cross_validation as cv

batch_size = 5
nb_classes = 1
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 80
# size of pooling area for max pooling
nb_pool = 2


train_data = np.load("train_data.npy")
target = np.load("train_target.npy")

X_train, X_test, Y_train, Y_test = cv.train_test_split(train_data, target, test_size=0.2, random_state=23)

model = Sequential()


model.add(Convolution2D(nb_filters, 7, 7,
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))
model.add(Activation('sigmoid'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Convolution2D(nb_filters, 4, 4))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Convolution2D(nb_filters, 4, 4))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Flatten())
model.add(Dense(300))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])