import pickle
import warnings

from keras import callbacks as ckbs
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


class FullModelCheckpoint(ckbs.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    pickle.dump([self.model.to_json(), self.model.get_weights()], open(filepath, 'wb'))
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            pickle.dump([self.model.to_json(), self.model.get_weights()], open(filepath, 'wb'))

class LossHistory(ckbs.Callback):
    def __init__(self, X_train, y_train, X_test, y_test, fp, score_func):
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        self.fp = fp
        self.score_func = score_func

    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []
        self.add_val_scores = []
        self.add_train_scores = []
        self.best_score = 0 
        
    def printCurrentStage(self, epoch):
        fps = self.fp.split("/")
        file = open("/".join(fps[:len(fps)-1]) + "/currentStage.txt", "w")
        file.write("params = %s, epoch = %d, val_score = %f" % (fps[len(fps)-1], epoch, self.best_score))
        file.close()

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(mse(self.y_train, self.model.predict(self.X_train)))
        self.val_losses.append(logs.get('val_loss'))        

        if self.score_func == 'accuracy':
            true_train = np_utils.probas_to_classes(self.y_train)
            pred_train = np_utils.probas_to_classes(self.model.predict(self.X_train))
            self.add_train_scores.append(accuracy_score(true_train, pred_train))

            true_test = np_utils.probas_to_classes(self.y_test)
            pred_test = np_utils.probas_to_classes(self.model.predict(self.X_test))
            val_score = accuracy_score(true_test, pred_test)
            self.add_val_scores.append(val_score)
        elif self.score_func == 'r2_score':
            val_score = r2_score(self.y_test, self.model.predict(self.X_test))
            self.add_val_scores.append(val_score)
            self.add_train_scores.append(r2_score(self.y_train, self.model.predict(self.X_train)))    
            
        self.best_score = max(self.best_score, val_score)
        self.printCurrentStage(epoch)