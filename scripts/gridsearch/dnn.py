import pickle
import pandas as pd
import numpy as np
import h5py
import theano
import os
from sklearn.metrics import r2_score
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from callbacks import LossHistory, FullModelCheckpoint

from keras.utils import np_utils
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from keras import callbacks as ckbs
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam, Adamax, SGD, RMSprop
from keras.layers.advanced_activations import PReLU

class DNNGridSearch():
    def __init__(self, params, log_files_path, output_dim=1, output_activation=None,
                 batch_size=16, epoch_count=1, score_func = 'r2_score', patience=150,
                 class_mode='categorical', dropout=0.5, loss='categorical_crossentropy', 
                 default_activation='relu', default_optimizer = Adadelta()):
        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.log_files_path = log_files_path
        self.createDirectoryIfNotExist(self.log_files_path)
        self.param_grid = ParameterGrid(params)
        self.patience = patience
        self.dropout = dropout
        self.score_func = score_func
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.default_activation = default_activation
        self.default_optimizer = default_optimizer
        self.loss = loss
        self.class_mode = class_mode
        
    def createDirectoryIfNotExist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
    def getFileName(self, hLayers, act, opt):
        hPrefix = "_".join([str(x) for x in hLayers])
        if (not isinstance(act, str)):
            str_act = act.__class__.__name__.lower()
        if (not isinstance(opt, str)):
            str_opt = opt.__class__.__name__.lower()
        return hPrefix + '_' + str_act + '_' + str_opt
       
        
    def createDNN(self, hLayers, activation, optimizer):
        dnn = Sequential()
        dnn.add(Dense(hLayers[0], input_dim = self.input_dim, W_regularizer=l2(0.01)))
        dnn.add(Activation(activation))
        if (self.dropout != 0):
            dnn.add(Dropout(self.dropout))
        for hl in hLayers[1:]:
            dnn.add(Dense(hl, W_regularizer=l2(0.01)))
            dnn.add(Activation(activation))
            if (self.dropout != 0):
                dnn.add(Dropout(self.dropout))
        dnn.add(Dense(self.output_dim))
        if (self.output_activation != None):
            dnn.add(Activation(self.output_activation))
        dnn.compile(loss=self.loss, optimizer=optimizer, class_mode=self.class_mode)
        return dnn
    
    def createCallBacks(self, X_train, y_train, X_test, y_test, filePath):
        lhae = LossHistory(X_train, y_train, X_test, y_test, filePath, self.score_func)
        checkpointer = FullModelCheckpoint(monitor='val_loss', filepath=filePath+'.insnn', verbose=0, save_best_only=True)
        es = ckbs.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0)
        return lhae, checkpointer, es
    
    def writeInfoToFile(self, lhae, filePath):
        df = pd.DataFrame()
        df['train_loss'] = lhae.train_losses
        df['val_loss'] = lhae.val_losses
        df['add_val_score'] = lhae.add_val_scores
        df['add_train_score'] = lhae.add_train_scores
        df.to_csv(filePath+'.csv', sep=',',index=False)
        
    def convertToArray(self, X):
        if (isinstance(X, pd.DataFrame)):
            X = X.get_values()
        return X
    
    def check_keys(self, valid_keys):
        for key in self.param_grid.param_grid[0].keys():
            if (key not in valid_keys):
                raise Exception("'%s' is invalid parameter" % (key))
    
    def fit(self, X_train, y_train, X_test, y_test):
        self.input_dim = X_train.shape[1]
        X_train = self.convertToArray(X_train)
        X_test = self.convertToArray(X_test)
        y_train = self.convertToArray(y_train)
        y_test = self.convertToArray(y_test)
        print("Start training. Configurations count: %i" % (len(self.param_grid)))
        i = 1
        best_score = 0
        self.check_keys(['hLayers','activation','optimizer'])
        for curParams in self.param_grid:
            if ('hLayers' not in curParams.keys()):
                raise Exception("'hLayers' must be one of the grid parameters")
            else:
                hLayers = curParams.get('hLayers')
                
            if ('activation' not in curParams.keys()):
                activation = self.default_activation
            else:
                activation = curParams.get('activation')
                
            if ('optimizer' not in curParams.keys()):
                optimizer = self.default_optimizer
            else:
                optimizer = curParams.get('optimizer')
                
            filePath = self.log_files_path + self.getFileName(hLayers, activation, optimizer)
            dnn = self.createDNN(hLayers, activation, optimizer)
            lhdnn, chkp, es = self.createCallBacks(X_train, y_train, X_test, y_test, filePath)
            dnn.fit(X_train, y_train, nb_epoch=self.epoch_count, batch_size=self.batch_size,
                   validation_data = [X_test, y_test], verbose=2, callbacks=[lhdnn, chkp, es])
            if self.score_func == 'accuracy':

                true_test = np_utils.probas_to_classes(y_test)
                pred_test = np_utils.probas_to_classes(dnn.predict(X_test))
                cur_score = accuracy_score(true_test, pred_test)
            
                if (cur_score > best_score):
                    best_score = cur_score
                    best_dnn = dnn
                self.writeInfoToFile(lhdnn, filePath)
                
            elif self.score_func == 'r2_score':
                cur_score = r2_score(y_test, dnn.predict(X_test))
            
                if (cur_score > best_score):
                    best_score = cur_score
                    best_dnn = dnn

                self.writeInfoToFile(lhdnn, filePath)

            print("Configuration #%i. Completed." % (i))
            i += 1
        return dnn
