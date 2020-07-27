# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import logging
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def do_dnn_1d(x, y, save_path, name, Input_shape=None):
    """

    Args:
        x: the data of train
        y: the labels of train
        save_path: the save path of model
        name: the name of model

    Returns:

    """
    print ("DNN and 1d")
    y = to_categorical(y, num_classes=5)
    # Building deep neural network
    input_layer = Input(shape=(Input_shape,))
    dense1 = Dense(64, activation='tanh')(input_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='tanh')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    predictions = Dense(5, activation='softmax')(dropout2)

    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = optimizers.SGD(lr=0.1,decay=0.96)

    # Training
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.fit(x, y, epochs=10, validation_split=0.3)

    # create HDF5 file
    model.save(save_path + name + 'model.h5')


