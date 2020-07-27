# data preprocessing
import logging
# Training
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical


def do_cnn_lstm(trainX, testX, trainY, testY):
    logging.info ("cnn_lstm")

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=5)
    testY = to_categorical(testY, num_classes=5)

    # Building cnn network
    inputs = Input(shape=(33,1), name='input')
    cnn_network = Conv1D(32, 3, padding='valid', activation='relu')(inputs)
    cnn_network = MaxPooling1D(pool_size=2)(cnn_network)
    cnn_network = Conv1D(64, 3, padding='valid', activation='relu')(cnn_network)
    cnn_network = MaxPooling1D(pool_size=2)(cnn_network)
    cnn_network = Dropout(0.5)(cnn_network)
    cnn_network = Flatten()(cnn_network)
    full_connection = Dense(33, activation='relu')(cnn_network)

    # Building lstm network
    lstm_network = Embedding(1000, 128)(full_connection)
    lstm_network = LSTM(128, dropout=0.8)(lstm_network)
    predictions = Dense(5, activation='softmax')(lstm_network)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=10,epochs=5)