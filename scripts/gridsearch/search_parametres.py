import pandas as pd
import random
import numpy as np
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from sklearn.preprocessing import MinMaxScaler
import theano
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adadelta, Adam, RMSprop, SGD
from keras.models import Sequential
from keras.regularizers import l1, l2
import _pickle as pickle
from keras.utils.np_utils import *
from sklearn.metrics import r2_score
from dnn import *
from callbacks import *
from utils import *
from autoencoder import *
import sklearn.cross_validation as cv
from sklearn.cross_validation import KFold


path_to_project = "C:/workspace/ml/graduate_work/vehicle_detection/" #windows"


train_data = np.load(path_to_project + "data/processed/train_data_new.npy")
target = np.load(path_to_project + "data/processed/train_target_new.npy")

X_train_comp, X_test_comp, y_train_comp, y_test_comp = cv.train_test_split(train_data, target, test_size=0.2, random_state=23)

#DNN Example
dnn_params = {'hLayers':[[900],[700,350],[700,350,120],[700,500,300,100]], 
'activation':['relu']}
dnn_gs = DNNGridSearch(params = dnn_params, epoch_count=200, log_files_path="BestModels/DNNs/")
dnn_gs.fit(X_train_comp, y_train_comp, X_test_comp, y_test_comp)

#big test
dnn_params = {'hLayers':[[800],[1000],[600,400],[700,350],[800,300],[1000,400],[700,350,150],[800,400,200]], 
				'activation':['relu','tanh','sigmoid', 'tanh'],'optimizer':[Adadelta(),Adam(),RMSprop(), SGD()]}

dnn_gs = DNNGridSearch(params = dnn_params, epoch_count=2500, patience=200, log_files_path="DNNTests/200116/big_test/")

dnn_gs.fit(X_train_comp, y_train_comp, X_test_comp, y_test_comp)

