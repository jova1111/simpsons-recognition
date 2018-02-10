import numpy as np
import heapq
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout,Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import load_model


def create_model():
    """
    Implementacija vestacke neuronske mreze sa 12288 (64x64x3) neurona na uloznom sloju,
    128 neurona u skrivenom sloju i 18 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18))
    model.add(Activation('softmax'))
    return model


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
    ann.fit(x_train, y_train, epochs=200, batch_size=32, verbose=1, shuffle=False)

    return ann


def create_network(ready_regions, region_data):
    """
    Trenira neuronsku mrezu i cuva je na disk.

    :param ready_regions: Regioni slika koji se koriste za ulaz u mrezu.
    :param region_data: Indeksi likova koji predstavljaju zeljeni izlaz iz mreze
    :return:
    """
    ann = create_model()
    ann = train_ann(ann, ready_regions, to_categorical(region_data))
    ann.save('neural_network_roi_dilated.h5')


def get_result(outputs):  # output je vektor sa izlaza neuronske mreze
    """
    Pronaci i vratiti indekse 5 neurona koji su najvise pobudjeni

    :param outputs: rezultati iz mreze (neuroni)
    :return: indeksi 5 najvise pobudjenih neurona
    """
    arr = outputs[0]

    return heapq.nlargest(5, range(len(arr)), arr.__getitem__)
