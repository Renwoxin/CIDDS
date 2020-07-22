from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    df_ = pd.read_csv(path,low_memory=False)

    # df_ = path

    # Flags one-hot
    Flags_ = pd.get_dummies(df_["Flags"].replace('nan', np.nan))
    df_Flags = pd.DataFrame(Flags_)

    # attackType one-hot
    attackType_ = pd.get_dummies(df_["attackType"].replace('nan', np.nan))
    df_attackType = pd.DataFrame(attackType_)

    df_['newBytes'] = get_features_Mb2bytes(df_['Bytes'])

    df = pd.concat([df_, df_Flags, df_attackType], axis=1, sort=False)


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