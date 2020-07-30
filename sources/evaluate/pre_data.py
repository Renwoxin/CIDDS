from sources.preprocess.features import *

def features_FPPB(path):
    """

    Args:
        path:

    Returns:

    """
    X, Y = get_features_FPPB(path)

    return X, Y


def features_hash(path):
    """

    Args:
        path:

    Returns:

    """
    X, Y = get_features_hash(path)

    return X, Y


def features_undersampling(path):
    """

    Args:
        path:

    Returns:

    """
    X, Y = get_features_undersampling(path)

    return X, Y

def features_unique(path_001, path_002, data_path, save_path):
    """

    Args:
        path:

    Returns:

    """
    Flags_dic, Proto_dic = gen_unique_dic(path_001, path_002, save_path)
    # extract_trd_data(path_001, path_002, save_path)
    gen_unique_data(data_path, Flags_dic, Proto_dic, save_path)

def features(path):
    """

    Args:
        path:

    Returns:

    """
    get_features(path)


def feature(path):
    """

    Args:
        path:

    Returns:

    """
    get_feature(path)