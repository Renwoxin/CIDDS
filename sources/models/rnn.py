# Training
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def do_rnn_wordbag(trainX, testX, trainY, testY,max_document_length):
    print("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=5)
    testY = to_categorical(testY, num_classes=5)

    # Network building
    inputs = Input((max_document_length,))
    net = Embedding(10240000, 128)(inputs)
    net = LSTM(128, dropout=0.8)(net)
    predictions = Dense(3, activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=1, epochs=5)