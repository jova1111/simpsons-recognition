import numpy as np
import operator
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import load_model


def create_ann():
    """
    Implementacija vestacke neuronske mreze sa 2352 neurona na uloznom sloju,
    128 neurona u skrivenom sloju i 18 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    """
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(18, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train):
    """
    Obucavanje vestacke neuronske mreze
    :param ann: mreza koja se obucava
    :param x_train: ulazne vrednosti
    :param y_train: izlazne vrednosti
    :return: obucena mreza
    """
    x_train = np.array(x_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(x_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann


def winner(outputs):  # output je vektor sa izlaza neuronske mreze
    """
    pronaci i vratiti indeks neurona koji je najvise pobudjen
    """
    return max(enumerate(outputs[0]), key=operator.itemgetter(1))


def get_result(outputs):
    """
    Vraca indeks lika za kojeg se dobila najveca vrednost u pogadjanju
    """
    index, value = winner(outputs)
    return index
