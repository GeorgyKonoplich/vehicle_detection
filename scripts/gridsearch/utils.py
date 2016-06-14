import pickle
from keras.models import model_from_json
from keras import callbacks as ckbs
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, AutoEncoder, Dropout
from keras.optimizers import Adagrad, Adadelta, Adam

def save_neural_network(nn, save_to): 
    w_path = ".".join(save_to.split(".")[:-1]) + ".hdf5"
    pickle.dump([nn.to_json(), w_path], open(save_to, 'wb'))
    nn.save_weights(w_path, overwrite=True)
    
def load_neural_network(file_from):
    (nn_arch, nn_weights_path) = pickle.load(open(file_from, 'rb'))
    nn = model_from_json(nn_arch)
    nn.set_weights(nn_weights_path)
    return nn

def get_model_architecture(file_from):
    (nn_arch, nn_weights_path) = pickle.load(open(file_from, 'rb'))
    return nn_arch
                    
def encode_features(autoencoder_path, X):
    encoder = get_encoder(autoencoder_path)
    arch = get_model_architecture(autoencoder_path)
    arch = json.loads(arch)
    ae = Sequential()
    ae.add(encoder)
    ae.compile(loss = arch['loss'],optimizer = arch['optimizer']['name'])
    return ae.predict(X)

def get_encoder(autoencoder_path):
    autoencoder = load_neural_network(autoencoder_path)
    return autoencoder.layers[0].encoder
