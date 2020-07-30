from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import mechine_learning
import numpy as np
import json
import os


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



if __name__=='__main__':
    path_001 = '/home/liyulian/data/CIDDS/CIDDS-001/traffic'
    path_002 = '/home/liyulian/data/CIDDS/CIDDS-002/traffic'
    save_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/'
    data_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/data_features_test_10000_35.csv'
    dic1_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/Flags_dic.npy'
    dic2_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/Proto_dic.npy'

    Flags_dic, Proto_dic = gen_unique_dic(path_001, path_002, save_path)
    # extract_trd_data(path_001, path_002, save_path)
    gen_unique_data(data_path,  Flags_dic, Proto_dic, save_path)