import theano
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.optimizers import Adadelta
import numpy as np
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU, ParametricSoftplus, ThresholdedReLU


class DenoisingAutoEncoder():
    def __init__(self, numpy_rng, hidden_layers = [],  theano_rng=None, metrics=['accuracy'], batch_size=16, nb_epoch=1,
                 loss='mean_squared_error', activation='relu', optimizer=Adadelta(), callbacks=[], corruption_level = 0.,verbose=1):

        self.corruption_level = corruption_level
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = K.RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        self.hidden_layers = hidden_layers
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.metrics = metrics
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.encoders = []
        self.acts = []

    def _get_act_by_name(self, act):
        str_act = ['relu', 'tanh', 'sigmoid', 'linear', 'softmax', 'softplus', 'softsign', 'hard_sigmoid']
        if (act in str_act):
            return Activation(act)
        else:
            return {'prelu': PReLU(), 'elu': ELU(), 'srelu': SReLU(), 'lrelu': LeakyReLU(),
                    'psoftplus': ParametricSoftplus(), 'trelu': ThresholdedReLU()}[act]

    def _get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def _get_model(self, input_shape):
        inputs = Input(shape=(input_shape,))
        state = inputs
        for n_out in self.hidden_layers:
            encoder = Dense(n_out)
            self.encoders.append(encoder)
            act = self._get_act_by_name(self.activation)
            self.acts.append(act)
            encoded = act(encoder(state))
            state = encoded

        list = self.hidden_layers[:-1]
        for n_out in reversed(list):
            act = self._get_act_by_name(self.activation)
            decoder = act(Dense(n_out)(state))
            state = decoder
        decoder = Dense(input_shape)(state)
        ae = Model(input=inputs, output=decoder)
        return ae

    def fit(self, X_train, X_test, y_train, y_test):
        self.ae = self._get_model(X_train.shape[1])
        self.ae.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        X_train = self._get_corrupted_input(X_train, self.corruption_level)
        X_train = X_train.eval()
        X_test = self._get_corrupted_input(X_test, self.corruption_level)
        X_test = X_test.eval()

        self.ae.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=self.batch_size,
                    nb_epoch=self.nb_epoch, verbose=self.verbose, callbacks=self.callbacks)

    def predict(self, X):
        return self.ae.predict(X)

    def transform(self, X, num):
        x = np.float32(X)
        for i in range(0, num + 1):
            encoder = self.encoders[i]
            act = self.acts[i]
            x = act(encoder(x)).eval()
        return x


