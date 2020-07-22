# -*- coding: utf-8 -*-
"""
    This is a demo of test, where  Credit Card Fraud Detection is used
"""

# data preprocessing
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

# Training
import xgboost as xgb
from sklearn import metrics
from tensorflow.keras.layers import Input,Dropout,Dense,Embedding,LSTM
from tensorflow.keras.layers import Conv1D,GlobalMaxPool1D
from tensorflow.keras.layers import Conv2D,MaxPool2D,Add,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features=50
max_document_length = 50

def read_allcsvdata(base_path1,base_path2):
    """

    Args:
        base_path:

    Returns:

    """
    df_all = pd.DataFrame([])

    # ExternalServer = os.listdir(os.path.join(base_path1, 'ExternalServer'))
    # for filename in (sorted(ExternalServer)):
    #     train_path = os.path.join(base_path1, 'ExternalServer', filename)
    #     df = pd.read_csv(train_path,low_memory=False)
    #     df_all = df_all.append(df)
    #
    OpenStack = os.listdir(os.path.join(base_path1, 'OpenStack'))
    for filename in (sorted(OpenStack)):
        train_path = os.path.join(base_path1, 'OpenStack', filename)
        df = pd.read_csv(train_path,low_memory=False)
        df_all = df_all.append(df)

    # traffic = os.listdir(base_path2)
    # for filename in sorted(traffic):
    #     train_path = os.path.join(base_path2, filename)
    #     df = pd.read_csv(train_path,low_memory=False)
    #     df_all = df_all.append(df)

    df_all.to_csv('data_features_001_O.csv')

    # return df_all

def data_StandardScaler(data):
    """

    Args:
        data: Data that needs to be standardized

    Returns: Standardized data

    """

    data = StandardScaler().fit_transform(data.values.reshape(-1, 1))
    return data


def get_features_by_onehot(basic_features):
    """

    Args:
        basic_features: Original features without any preprocessing

    Returns:
         features: Features of Bag of Words Vectorization

    """
    vectorizer = CountVectorizer(ngram_range=(2, 4),
                                 token_pattern=r'\w',
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)

    features=vectorizer.fit_transform(basic_features)
    features = features.toarray()
    return features

def get_features_Mb2bytes(basic_features):
    """

    Args:
        basic_features:

    Returns:

    """
    features = basic_features.values.tolist()
    for i in range(len(features)):
        if features[i].split()[-1] == "M":
            features[i] = str(int(float(features[i].split()[0])*1024*1024))

    features = np.array(features)
    return features

def get_features(path):
    """

    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        X_test: DataFrame,shape=(x.shape[0] * test_size,x.shape[1]), the features for testing
        Y_train: list:x.shape[0] * (1-test_size), the labels for training
        Y_test: list:x.shape[0] * test_size, the labels for testing

    """

    # df = pd.read_csv(path)

    df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags], axis=1, sort=False)


    # newBytes = pd.DataFrame(Bytes)
    # df = df.join(newBytes)

    # df['norm_Src_Pt'] = data_StandardScaler(df['Src Pt'])
    # df['norm_Dst_Pt'] = data_StandardScaler(df['Dst Pt'])
    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    #
    # lbl = preprocessing.LabelEncoder()
    # df['acc_id1'] = lbl.fit_transform(train_x['acc_id1'].astype(str))  # 将提示的包含错误数据类型这一列进行转换


    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Src Pt', 'Dst Pt', 'Packets', 'Src IP Addr',
                  'Dst IP Addr', 'Bytes', 'Flags','attackType',	'attackID',	'attackDescription', 'newBytes'], axis=1)

    normal_index = df[df['class'] == 'normal'].index
    attacker_index = df[df['class'] == 'attacker'].index
    victim_index = df[df['class'] == 'victim'].index
    suspicious_index = df[df['class'] == 'suspicious'].index
    unknown_index = df[df['class'] == 'unknown'].index

    x_index = np.concatenate([normal_index,attacker_index,victim_index,suspicious_index,unknown_index])
    df = df.drop(['class'],axis=1)

    x = df.iloc[x_index,:]
    y = [0] * len(normal_index) + [1] * len(attacker_index) + [2] * len(victim_index) + [3] * len(suspicious_index) + [4] * len(unknown_index)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)
    return X_train, X_test, Y_train, Y_test

def do_xgboost(X_train, X_test, Y_train, Y_test):
    """

    Args:
        X_train:DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        X_test: DataFrame, shape(x_index*test_size, feature_number(feature_map)), the features for testing
        Y_train: list, the labels for training
        Y_test: list, the labels for testing

    Returns:None

    """
    xgb_model = xgb.XGBClassifier().fit(X_train, Y_train)
    Y_pred = xgb_model.predict(X_test)
    do_metrics(Y_test, Y_pred)

def do_metrics(Y_test,Y_pred):
    """

    Args:
        Y_test: list, the labels for testing
        Y_pred: list, the labels for prediction

    Returns:None

    """
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(Y_test, Y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(Y_test, Y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(Y_test, Y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(Y_test, Y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(Y_test,Y_pred))



def do_rnn_wordbag(trainX, testX, trainY, testY):
    print ("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, num_classes=5)
    testY = to_categorical(testY, num_classes=5)

    # Network building
    inputs = Input((max_document_length,))
    net = Embedding(200,128)(inputs)
    net = LSTM(128, dropout=0.8)(net)
    predictions = Dense(5, activation='softmax')(net)

    # Training
    adam = optimizers.Adam(lr=0.001)
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=1,epochs=5)



if __name__=="__main__":
    # base_path1 = '/home/liyulian/data/CIDDS/CIDDS-001/traffic'
    # base_path2 = '/home/liyulian/data/CIDDS/CIDDS-002/traffic'
    # path = '/home/liyulian/data/CIDDS/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week1.csv'

    # 单独使用base_path1中OpenStack的数据集且提前生成好了
    # path = read_allcsvdata(base_path1, base_path2)

    path = 'data_features_001_E.csv'
    # Flags特征使用的是one-hot方式，没有用Proto、Src Pt和Dst Pt三个个特征
    X_train, X_test, Y_train, Y_test = get_features(path)
    do_xgboost(X_train, X_test, Y_train, Y_test)
    # do_rnn_wordbag(np.array(X_train), np.array(X_test), Y_train, Y_test)