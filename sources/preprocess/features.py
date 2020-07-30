from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
import logging
import os
import json


def data_StandardScaler(data):
    """

    Args:
        data: Data that needs to be standardized

    Returns: Standardized data

    """

    data = StandardScaler().fit_transform(data.values.reshape(-1, 1))
    return data


def get_features_Mb2bytes(basic_features):
    """

    Args:
        basic_features:

    Returns:

    """
    features = basic_features.values.tolist()
    for i in range(len(features)):
        if str(features[i]).split()[-1] == "M":
            features[i] = str(int(float(features[i].split()[0])*1024*1024))

    features = np.array(features)
    return features

def hash_features(basic_features):
    """

    Args:
        basic_features:

    Returns:

    """

    h = FeatureHasher(n_features=20, input_type='string', dtype=int, alternate_sign=False)
    features_ = [str(basic_features.values.tolist()[i]) for i in range(len(basic_features.values.tolist()))]
    features = h.fit_transform(features_)

    return features


def get_features(path):
    """
    get_features:Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features are used,
                 Flags and Proto features use one-hot mode,
                 and Bytes features are preprocessed
    Args:
        path: the path of Dataset

    Returns:
        x: DataFrame, shape=(x.shape[0], x.shape[1]), the features for training
        y: list:x.shape[0], the labels for training

    """
    logging.info('get_features:'
                 'Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features are used,'
                 'Flags and Proto features use one-hot mode, '
                 'and Bytes features are preprocessed')


    df_ = pd.read_csv(path,low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # Proto one-hot
    Proto_ = pd.get_dummies(df_["Proto"].replace('nan', np.nan))
    df_Proto = pd.DataFrame(Proto_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags, df_Proto], axis=1, sort=False)

    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Packets', 'Src IP Addr',
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

    return x,y

def get_features_FPPB(path):
    """
    get_features_FPPB:Flags, Proto, Packets, Bytes features are used,
                      Flags and Proto two types of features use one-hot mode,
                      and Bytes features are preprocessed
    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        X_test: DataFrame,shape=(x.shape[0] * test_size,x.shape[1]), the features for testing
        Y_train: list:x.shape[0] * (1-test_size), the labels for training
        Y_test: list:x.shape[0] * test_size, the labels for testing

    """
    logging.info('get_features_FPPB:'
                         'Flags, Proto, Packets, Bytes features are used,'
                         'Flags and Proto two types of features use one-hot mode,'
                         'and Bytes features are preprocessed')

    df_ = pd.read_csv(path,low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # Proto one-hot
    Proto_ = pd.get_dummies(df_["Proto"].replace('nan', np.nan))
    df_Proto = pd.DataFrame(Proto_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags, df_Proto], axis=1, sort=False)

    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Packets', 'Src IP Addr', 'Src Pt', 'Dst Pt',
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

    return x, y


def get_features_undersampling(path):
    """
    get_features_undersampling:This function downsampling is for data set 002,
                               using Flags, Proto, Packets, Bytes features,
                               Flags, Proto two types of features using one-hot mode,
                               Bytes feature preprocessing
    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        X_test: DataFrame,shape=(x.shape[0] * test_size,x.shape[1]), the features for testing
        Y_train: list:x.shape[0] * (1-test_size), the labels for training
        Y_test: list:x.shape[0] * test_size, the labels for testing

    """
    logging.info('get_features_undersampling:'
                         'This function downsampling is for data set 002,'
                         'using Flags, Proto, Packets, Bytes features,'
                         'Flags, Proto two types of features using one-hot mode,'
                         'Bytes feature preprocessing')

    df_ = pd.read_csv(path,low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # Proto one-hot
    Proto_ = pd.get_dummies(df_["Proto"].replace('nan', np.nan))
    df_Proto = pd.DataFrame(Proto_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags, df_Proto], axis=1, sort=False)

    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Packets', 'Src IP Addr', 'Src Pt', 'Dst Pt',
                  'Dst IP Addr', 'Bytes', 'Flags','attackType',	'attackID',	'attackDescription', 'newBytes'], axis=1)

    normal_index = df[df['class'] == 'normal'].index
    attacker_index = df[df['class'] == 'attacker'].index
    victim_index = df[df['class'] == 'victim'].index
    suspicious_index = df[df['class'] == 'suspicious'].index
    unknown_index = df[df['class'] == 'unknown'].index

    random_choice_normal_index = np.random.choice(normal_index, size=len(victim_index), replace=False)
    random_choice_attacker_index = np.random.choice(attacker_index, size=len(victim_index), replace=False)

    x_index = np.concatenate([random_choice_normal_index, random_choice_attacker_index, victim_index])

    df = df.drop(['class'],axis=1)

    x = df.iloc[x_index,:]
    y = [0] * len(victim_index) + [1] * len(victim_index) + [2] * len(victim_index)


    return x, y


def get_features_FPB(path):
    """
    get_features_FPB:This function is not down-sampling, using Flags, Packets, Bytes features,
                     Flags features using one-hot,
                     and Bytes features preprocessing
    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        X_test: DataFrame,shape=(x.shape[0] * test_size,x.shape[1]), the features for testing
        Y_train: list:x.shape[0] * (1-test_size), the labels for training
        Y_test: list:x.shape[0] * test_size, the labels for testing

    """
    logging.info('get_features_FPB:This function is not down-sampling, using Flags, Packets, Bytes features,'
                     'Flags features using one-hot,'
                     'and Bytes features preprocessing')
    df_ = pd.read_csv(path,low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # Proto one-hot
    # Proto_ = pd.get_dummies(df_["Proto"].replace('nan', np.nan))
    # df_Proto = pd.DataFrame(Proto_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags], axis=1, sort=False)

    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Packets', 'Src IP Addr', 'Src Pt', 'Dst Pt',
                  'Dst IP Addr', 'Bytes', 'Flags','attackType',	'attackID',	'attackDescription', 'newBytes'], axis=1)

    normal_index = df[df['class'] == 'normal'].index
    attacker_index = df[df['class'] == 'attacker'].index
    victim_index = df[df['class'] == 'victim'].index
    suspicious_index = df[df['class'] == 'suspicious'].index
    unknown_index = df[df['class'] == 'unknown'].index

    random_choice_normal_index = np.random.choice(normal_index, size=len(victim_index), replace=False)
    random_choice_attacker_index = np.random.choice(attacker_index, size=len(victim_index), replace=False)

    x_index = np.concatenate([random_choice_normal_index, random_choice_attacker_index, victim_index])

    df = df.drop(['class'],axis=1)

    x = df.iloc[x_index,:]
    y = [0] * len(victim_index) + [1] * len(victim_index) + [2] * len(victim_index)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)

    return X_train, X_test, Y_train, Y_test

def get_features_train_test(path):
    """
    get_features:Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features are used,
    Flags and Proto features use one-hot mode, and Bytes features are preprocessed
    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        Y_train: list:x.shape[0] * (1-test_size), the labels for training

    """
    logging.info('get_features:'
                        'Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features are used,'
                        'Flags and Proto features use one-hot mode, and Bytes features are preprocessed')
    df_ = pd.read_csv(path,low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # Proto one-hot
    Proto_ = pd.get_dummies(df_["Proto"].replace('nan', np.nan))
    df_Proto = pd.DataFrame(Proto_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags, df_Proto], axis=1, sort=False)

    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Packets', 'Src IP Addr',
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

    return x,y

def get_features_hash(path):
    """
    get_features:Use Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features,
                 Flags, Proto two types of features use one-hot method,
                 Bytes feature is preprocessed, Src Pt, Dst Pt features are hashed
    Args:
        path: the path of Dataset

    Returns:
        X_train: DataFrame, shape=(x.shape[0] * (1-test_size),x.shape[1]), the features for training
        X_test: DataFrame,shape=(x.shape[0] * test_size,x.shape[1]), the features for testing
        Y_train: list:x.shape[0] * (1-test_size), the labels for training
        Y_test: list:x.shape[0] * test_size, the labels for testing

    """
    logging.info('get_features:'
                        'Use Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features,'
                        'Flags, Proto two types of features use one-hot method,'
                        'Bytes feature is preprocessed, Src Pt, Dst Pt features are hashed')

    df_ = pd.read_csv(path, low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # Proto one-hot
    Proto_ = pd.get_dummies(df_["Proto"].replace('nan', np.nan))
    df_Proto = pd.DataFrame(Proto_)

    # hashfeature
    df_['Src_Pt'] = hash_features(df_['Src Pt'])

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags, df_Proto], axis=1, sort=False)

    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Proto', 'Flows', 'Tos', 'Packets', 'Src IP Addr', 'Src Pt', ' Dst Pt'
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

    return x, y


def get_feature(path):
    """
    get_feature:for unique
    Args:
        path: the path of Dataset

    Returns:
        x: DataFrame, shape=(x.shape[0], x.shape[1]), the features for training
        y: list:x.shape[0], the labels for training

    """
    logging.info('get_features:'
                 'Flags, Proto, Packets, Bytes, Src Pt, Dst Pt features are used,'
                 'Flags and Proto features use one-hot mode, '
                 'and Bytes features are preprocessed')


    df = pd.read_csv(path,low_memory=False)

    # df_ = path

    df['newBytes'] = get_features_Mb2bytes(df['Bytes'])
    df['norm_Packets'] = data_StandardScaler(df['Packets'])
    df['norm_newBytes'] = data_StandardScaler(df['newBytes'])

    df = df.drop(['Date first seen', 'Duration', 'Flows', 'Tos', 'Packets', 'Src IP Addr',
                  'Dst IP Addr', 'Bytes', 'attackType',	'attackID',	'attackDescription', 'newBytes'], axis=1)

    normal_index = df[df['class'] == 'normal'].index
    attacker_index = df[df['class'] == 'attacker'].index
    victim_index = df[df['class'] == 'victim'].index
    suspicious_index = df[df['class'] == 'suspicious'].index
    unknown_index = df[df['class'] == 'unknown'].index

    x_index = np.concatenate([normal_index,attacker_index,victim_index,suspicious_index,unknown_index])
    df = df.drop(['class'],axis=1)

    x = df.iloc[x_index,:]
    y = [0] * len(normal_index) + [1] * len(attacker_index) + [2] * len(victim_index) + [3] * len(suspicious_index) + [4] * len(unknown_index)

    return x,y


def save_jsonfile(path, item):
    # 先将字典对象转化为可写入文本的字符串
    item = json.dumps(item)

    try:
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                f.write(item + ",\n")
                print("^_^ write success")
        else:
            with open(path, "a", encoding='utf-8') as f:
                f.write(item + ",\n")
                print("^_^ write success")
    except Exception as e:
        print("write error==>", e)

def gen_unique_dic(path_001, path_002, save_path):
    """

    Args:
        path_001:
        path_002:

    Returns:

    """
    Flags_array = []

    Proto_array = []

    OpenStack = os.listdir(os.path.join(path_001, 'OpenStack'))
    for filename in (OpenStack):
        train_path = os.path.join(path_001, 'OpenStack', filename)
        df = pd.read_csv(train_path, low_memory=False)
        Flags_array_ = np.unique(df['Flags'].values)
        Flags_array.extend(Flags_array_)
        Proto_array_ = np.unique(df['Proto'].values)
        Proto_array.extend(Proto_array_)

    traffic = os.listdir(path_002)
    for filename in traffic:
        train_path = os.path.join(path_002, filename)
        df = pd.read_csv(train_path, low_memory=False)
        Flags_array_ = np.unique(df['Flags'].values)
        Flags_array.extend(Flags_array_)
        Proto_array_ = np.unique(df['Proto'].values)
        Proto_array.extend(Proto_array_)

    Flags_array = np.unique(Flags_array)
    Proto_array = np.unique(Proto_array)

    Flags_key = range(1, len(Flags_array)+1)
    Proto_key = range(1, len(Proto_array) + 1)

    Flags_dic = dict(zip(Flags_array, Flags_key))
    Proto_dic = dict(zip(Proto_array, Proto_key))

    # save_file(save_path +'Flags_dic.json',Flags_dic)
    # save_file(save_path + 'Proto_dic.json', Proto_dic)

    # np.save(save_path + 'Flags_dic.npy', Flags_dic)
    # np.save(save_path + 'Proto_dic.npy', Proto_dic)
    return Flags_dic, Proto_dic

def extract_trd_data(path_001, path_002, save_path):
    """

    Args:
        path_001:
        path_002:

    Returns:

    """
    x_train = pd.DataFrame([])
    x_test= pd.DataFrame([])
    OpenStack = os.listdir(os.path.join(path_001, 'OpenStack'))
    for filename in (OpenStack):
        train_path = os.path.join(path_001,'OpenStack', filename)
        df = pd.read_csv(train_path, low_memory=False)

        x_index_S_train = df[df['Src IP Addr'].values == '192.168.100.5'].index
        x_index_D_train = df[df['Dst IP Addr'].values == '192.168.100.5'].index

        x_index_S_test= df[df['Src IP Addr'].values == '10000_35'].index
        x_index_D_test = df[df['Dst IP Addr'].values == '10000_35'].index

        x_index_train = np.concatenate([x_index_S_train,x_index_D_train])
        x_train_ =df.iloc[x_index_train,:]
        x_train = x_train.append(x_train_)

        x_index_test = np.concatenate([x_index_S_test, x_index_D_test])
        x_test_ = df.iloc[x_index_test, :]
        x_test = x_train.append(x_test_)

    traffic = os.listdir(path_002)
    for filename in traffic:
        train_path = os.path.join(path_002, filename)
        df = pd.read_csv(train_path, low_memory=False)
        x_index_S_train = df[df['Src IP Addr'].values == '192.168.100.5'].index
        x_index_D_train = df[df['Dst IP Addr'].values == '192.168.100.5'].index

        x_index_S_test = df[df['Src IP Addr'].values == '10000_35'].index
        x_index_D_test = df[df['Dst IP Addr'].values == '10000_35'].index

        x_index_train = np.concatenate([x_index_S_train, x_index_D_train])
        x_train_ = df.iloc[x_index_train, :]
        x_train = x_train.append(x_train_)

        x_index_test = np.concatenate([x_index_S_test, x_index_D_test])
        x_test_ = df.iloc[x_index_test, :]
        x_test = x_train.append(x_test_)

    # x_train.to_csv(save_path + 'data_features_train.csv')
    x_test.to_csv(save_path + 'data_features_test_10000_35.csv')

def gen_unique_data(data_path, dic1, dic2,save_path):
    """

    Args:
        path:

    Returns:

    """
    df = pd.read_csv(data_path, low_memory=False)
    # with open(dic1_path, 'r') as f:
    #     dic1 = json.load(f)
    #
    # dic2 = json.loads(dic2_path)
    # dic1 = np.load(dic1_path)
    # dic2 = np.load(dic2_path)
    for i in range(len(df['Flags'].values)):
        df['Flags'].values[i] = dic1[df['Flags'].values[i]]

    for i in range(len(df['Proto'].values)):
        df['Proto'].values[i] = dic2[df['Proto'].values[i]]

    df.to_csv(save_path + 'data_features_test_unique_1000_35.csv')