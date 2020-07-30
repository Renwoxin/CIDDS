import os
import pandas as pd

def read_ciddds_csvdata(base_path1,base_path2, save_path, E_Server=False, O_Stack=False, t_raffic=False):
    """

    Args:
        base_path1: the path of CIDDS-001
        base_path2: the path of CIDDS-002
        save_path: the save path of csvdata
        E_Server: Control read 001_E data set
        O_Stack: Control read 001_O data set
        t_raffic: Control read 002 data set

    Returns:None

    """
    df_all = pd.DataFrame([])

    if E_Server==True:
        ExternalServer = os.listdir(os.path.join(base_path1, 'ExternalServer'))
        for filename in (sorted(ExternalServer)):
            train_path = os.path.join(base_path1, 'ExternalServer', filename)
            df = pd.read_csv(train_path, low_memory=False)
            df_all = df_all.append(df)

        df_all.to_csv(save_path + 'data_features_001_E.csv')

    elif O_Stack==True:
        OpenStack = os.listdir(os.path.join(base_path1, 'OpenStack'))
        for filename in (sorted(OpenStack)):
            train_path = os.path.join(base_path1, 'OpenStack', filename)
            df = pd.read_csv(train_path, low_memory=False)
            df_all = df_all.append(df)

        df_all.to_csv(save_path + 'data_features_001_O.csv')

    elif t_raffic==True:
        traffic = os.listdir(base_path2)
        for filename in sorted(traffic):
            train_path = os.path.join(base_path2, filename)
            df = pd.read_csv(train_path, low_memory=False)
            df_all = df_all.append(df)

        df_all.to_csv(save_path + 'data_features_002.csv')

    ExternalServer = os.listdir(os.path.join(base_path1, 'ExternalServer'))
    for filename in (sorted(ExternalServer)):
        train_path = os.path.join(base_path1, 'ExternalServer', filename)
        df = pd.read_csv(train_path, low_memory=False)
        df_all = df_all.append(df)

    OpenStack = os.listdir(os.path.join(base_path1, 'OpenStack'))
    for filename in (sorted(OpenStack)):
        train_path = os.path.join(base_path1, 'OpenStack', filename)
        df = pd.read_csv(train_path, low_memory=False)
        df_all = df_all.append(df)

    traffic = os.listdir(base_path2)
    for filename in sorted(traffic):
        train_path = os.path.join(base_path2, filename)
        df = pd.read_csv(train_path, low_memory=False)
        df_all = df_all.append(df)

    df_all.to_csv(save_path + 'data_features.csv')