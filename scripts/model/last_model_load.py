import pickle
from keras.models import model_from_json
import numpy as np
np.random.seed(1337)
from keras.optimizers import SGD, Adadelta
from sklearn.metrics import r2_score
import sklearn.cross_validation as cv


#path_to_project = "/home/konoplich/workspace/projects/BloodTranscriptome/scripts/data/vehicle_detection/" #ubuntu
path_to_project = "C:/workspace/ml/vehicle_detection/" #windows"

path_to_model = path_to_project + "/models/dnnnew";

path_to_train_data = path_to_project + "data/processed/train_data_new.npy";
path_to_target_data = path_to_project + "data/processed/train_target_new.npy";

train_data = np.load(path_to_train_data)
target = np.load(path_to_target_data)

X_train, X_test, Y_train, Y_test = cv.train_test_split(train_data, target, test_size=0.2, random_state=23)

def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pickle.load(open(file_from, 'rb'))
    nn = model_from_json(nn_arch)
    nn.set_weights(nn_weights_path)
    return nn


model = load_neural_network(path_to_model)

sgd = SGD(lr=0.001, decay=0, momentum=0, nesterov=True)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])

print(r2_score(Y_test, model.predict(X_test)))