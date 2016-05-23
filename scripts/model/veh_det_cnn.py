import pickle
from keras.models import model_from_json
import numpy as np
np.random.seed(1337)
import keras
from keras.optimizers import SGD, Adadelta 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import sklearn.cross_validation as cv

batch_size = 20
nb_classes = 1
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 80
# size of pooling area for max pooling
nb_pool = 2

path_to_project = "C:/workspace/ml/graduate_work/vehicle_detection/" #windows"

path_to_save_models = path_to_project + "/models/dnnnew"

train_data = np.load(path_to_project + "data/processed/train_data_new.npy")
target = np.load(path_to_project + "data/processed/train_target_new.npy")

X_train, X_test, Y_train, Y_test = cv.train_test_split(train_data, target, test_size=0.2, random_state=23)

model = Sequential()


model.add(Convolution2D(nb_filters, 7, 7,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Activation('sigmoid'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Convolution2D(nb_filters, 4, 4))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Convolution2D(nb_filters, 4, 4))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Flatten())
model.add(Dense(300))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

model.compile(loss='binary_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
def save_neural_network(nn, save_to):
    w_path = ".".join(save_to.split(".")[:-1]) + ".hdf5"
    pickle.dump([nn.to_json(), w_path], open(save_to, 'wb'))
    nn.save_weights(w_path, overwrite=True)
'''

def save_neural_network(nn, save_to):
    pickle.dump([nn.to_json(), nn.get_weights()], open(save_to, 'wb'))


save_neural_network(model, path_to_save_models)

