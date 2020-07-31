# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import logging
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sources.utils.get_history import LossHistory
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def do_dnn_1d(x, y, result_path, Input_shape=None):
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
    adam = optimizers.Adam(lr=1e-5, decay=0.96)

    # data
    X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.2, random_state=1)

    # Training
    model = Model(inputs=input_layer, outputs=predictions)

    model.summary()
    history = LossHistory()

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100,
              validation_data=(X_valid, Y_valid),
              batch_size=256,
              callbacks=[history])
    """
    早停，loss曲线，acc曲线，模型参数打印,history
    """
    history.loss_plot('epoch', result_path)
    return model


def do_dnn_1d_sample(x, y, result_path, Input_shape=None):
    """

    Args:
        x: the data of train
        y: the labels of train
        save_path: the save path of model
        name: the name of model

    Returns:

    """
    print("DNN and 1d")
    y = to_categorical(y, num_classes=5)
    # Building deep neural network
    input_layer = Input(shape=(Input_shape,))
    dense1 = Dense(32, activation='relu')(input_layer)
    # drop1 = Dropout(0.1)(dense1)
    # bn = BatchNormalization()(dense1)
    dense2 = Dense(60, activation='relu')(dense1)
    # dense3 = Dense(32, activation='relu')(dense2)
    # drop2 = Dropout(0.1)(dense3)
    predictions = Dense(5, activation='softmax')(dense2)

    # Regression using adam with learning rate decay and Top-3 accuracy
    adam = optimizers.Adam(lr=1e-4,decay=0.96)

    # data
    X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.2, random_state=1)

    # Training
    model = Model(inputs=input_layer, outputs=predictions)

    model.summary()
    history = LossHistory()
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100,
              validation_data=(X_valid, Y_valid),
              batch_size=128,
              callbacks=[history,early_stopping])
    """
    早停，loss曲线，acc曲线，模型参数打印, history
    """
    history.loss_plot('epoch', result_path)
    return model


