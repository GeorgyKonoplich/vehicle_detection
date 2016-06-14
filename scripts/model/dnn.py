import cPickle as pickle
from keras.models import model_from_json
import numpy as np
np.random.seed(1337)
import keras
from keras.optimizers import Adadelta 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import sklearn.cross_validation as cv
from sklearn.cross_validation import KFold


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

batch_size = 128
nb_classes = 1
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 80
# size of pooling area for max pooling
nb_pool = 2

path_to_project = "C:/workspace/ml/graduate_work/vehicle_detection/" #windows"

path_to_save_models = path_to_project + "/models/dnn200_128"

train_data = np.load(path_to_project + "data/processed/train_data_new.npy")
target = np.load(path_to_project + "data/processed/train_target_new.npy")

X_train, X_test, Y_train, Y_test = cv.train_test_split(train_data, target, test_size=0.2, random_state=23)

def create_model():
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
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid'))
	
	adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

	model.compile(loss='binary_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])

	return model


def save_neural_network(nn, save_to):
    pickle.dump([nn.to_json(), nn.get_weights()], open(save_to, 'wb'))


kf = KFold(len(train_data), n_folds=5, shuffle=True, random_state=23)
count = 0
acc = []

for train_index, test_index in kf:
	count += 1
	history = LossHistory()
	model = create_model()
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = train_data[train_index], train_data[test_index]
	y_train, y_test = target[train_index], target[test_index]
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          	verbose=1, validation_data=(X_test, y_test), callbacks=[history])
	his = np.array(history.losses)
	np.save("C:/workspace/ml/graduate_work/vehicle_detection/data/processed/history200_" + str(count), his)
	save_neural_network(model, path_to_save_models + str(count))
	score = model.evaluate(X_test, y_test, verbose=0)
	acc.append(score[1])

acc = np.array(acc)
print(acc)
np.save("C:/workspace/ml/graduate_work/vehicle_detection/data/processed/acc200_128", acc)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])




