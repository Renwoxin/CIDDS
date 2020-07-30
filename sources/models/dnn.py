# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import logging
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def do_dnn_1d(x, y, Input_shape=None):
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
    bn1 = BatchNormalization()(input_layer)
    dense1 = Dense(32, activation='relu')(bn1)
    bn2 = BatchNormalization()(dense1)
    dense2 = Dense(64, activation='relu')(bn2)
    predictions = Dense(5, activation='softmax')(dense2)

    # Regression using adam with learning rate decay and Top-3 accuracy
    adam = optimizers.Adam(lr=1e-5,decay=0.96)

    # data
    X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.2, random_state=1)

    # Training
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid))
    """
    早停，loss曲线，acc曲线，模型参数打印,history
    """
    return model


def do_dnn_1d_sample(x, y, Input_shape=None):
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
    bn1 = BatchNormalization()(input_layer)
    dense1 = Dense(32, activation='relu')(bn1)
    bn2 = BatchNormalization()(dense1)
    dense2 = Dense(64, activation='relu')(bn2)
    predictions = Dense(5, activation='softmax')(dense2)

    # Regression using adam with learning rate decay and Top-3 accuracy
    adam = optimizers.Adam(lr=1e-5,decay=0.96)

    # data
    X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.2, random_state=1)

    # Training
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid), batch_size=256, sample_weight=compute_sample_weight("balanced", Y_train))
    """
    早停，loss曲线，acc曲线，模型参数打印,history
    """
    return model


