import cPickle as pickle
from keras.models import model_from_json, Graph
import numpy as np
np.random.seed(1337)
import keras
from keras.optimizers import SGD, Adadelta 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import sklearn.cross_validation as cv
from sklearn.cross_validation import KFold
from keras.utils.np_utils import *

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 84
# size of pooling area for max pooling
nb_pool = 2

batch_size = 50
nb_classes = 2
nb_epoch = 200


path_to_project = "C:/workspace/ml/graduate_work/vehicle_detection/" #windows"

path_to_save_models = path_to_project + "/models/hdnn_classification_200_50"

train_data = np.load(path_to_project + "data/processed/train_data_classification.npy")
target = np.load(path_to_project + "data/processed/train_target_classification.npy")

target = to_categorical(target)
X_train, X_test, Y_train, Y_test = cv.train_test_split(train_data, target, test_size=0.2, random_state=23)

def save_neural_network(nn, save_to):
    pickle.dump([nn.to_json(), nn.get_weights()], open(save_to, 'wb'))

def create_model():
	model = Sequential()

	model.add(Convolution2D(nb_filters, 7, 7,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
	model.add(Activation('tanh'))
	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	
	model.add(Convolution2D(nb_filters, 4, 4))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

	model.add(Convolution2D(nb_filters, 4, 4))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

	model.add(Flatten())
	model.add(Dense(300))
	model.add(Activation('tanh'))
	model.add(Dense(nb_classes))
	model.add(Activation('tanh'))
	sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)

	model.compile(loss='categorical_crossentropy',
              optimizer=SGD,
              metrics=['accuracy'])

	return model

def create_graph_model():
	graph = Graph()

	graph.add_input(name='input', input_shape=(1, img_rows, img_cols))
    
	graph.add_node(Convolution2D(nb_filters, 7, 7, activation='tanh', border_mode='valid'), name='conv1', input='input')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool1', input='conv1')

	graph.add_node(Convolution2D(nb_filters, 4, 4, activation='tanh', border_mode='valid'), name='conv2', input='pool1')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool2', input='conv2')
	#hdnn part
	graph.add_node(Convolution2D(54, 6, 6, activation='tanh', border_mode='valid'), name='conv31', input='pool2')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool31', input='conv31')
	
	graph.add_node(Convolution2D(20, 4, 4, activation='tanh', border_mode='valid'), name='conv32', input='pool2')
	graph.add_node(MaxPooling2D(pool_size=(3, 3)), name='pool32', input='conv32')
	
	graph.add_node(Convolution2D(10, 2, 2, activation='tanh', border_mode='valid'), name='conv33', input='pool2')
	graph.add_node(MaxPooling2D(pool_size=(4, 4)), name='pool33', input='conv33')
	
	graph.add_node(Flatten(), name='flatten', inputs=['pool31', 'pool32', 'pool33'], merge_mode='concat', concat_axis=1)

	graph.add_node(Dense(300, activation='tanh'), name='dense1', input='flatten')
	graph.add_node(Dense(nb_classes, activation='tanh'), name='dense2', input='dense1')
	graph.add_output(name='output', input='dense2')

	sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)

	graph.compile(loss={'output':'categorical_crossentropy'},
              optimizer=sgd,
              metrics=['accuracy'])

	return graph

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in self.totals.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v / self.seen)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)



kf = KFold(len(train_data), n_folds=5, shuffle=True, random_state=23)
count = 0
acc = []

for train_index, test_index in kf:
	count += 1
	history = LossHistory()
	model = create_graph_model()
	X_train, X_test = train_data[train_index], train_data[test_index]
	y_train, y_test = target[train_index], target[test_index]
	model.fit({'input':X_train, 'output':y_train}, batch_size=batch_size, nb_epoch=nb_epoch,
          	verbose=1, validation_data=({'input':X_test, 'output':y_test}), callbacks=[history])

	his = np.array(history.losses)
	np.save("C:/workspace/ml/graduate_work/vehicle_detection/data/processed/history_hdnn_class_200_50" + str(count), his)
	save_neural_network(model, path_to_save_models + str(count))
	score = model.evaluate(X_test, y_test, verbose=0)
	acc.append(score[1])

accuracy = np.array(accuraccy)
print(accuracy)
np.save("C:/workspace/ml/graduate_work/vehicle_detection/data/processed/acc_hdnn_class_200_50", accuracy)




