from sources.utils.logfile import save_log_file
from sources.models import mechine_learning
from sources.models import dnn
import logging

def CIDDS_none(save_path, log_path,X_train, Y_train, name):
    """

    Args:
        save_path:
        log_path:
        X_train:
        Y_train:
        name:

    Returns:

    """
    save_log_file(log_path)

    ### 无样本平衡
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '无样本平衡, 设置了划分样本的random_state=1, 并且保存模型')

    model = mechine_learning.do_xgboost(X_train, Y_train)

    # create HDF5 file
    model.save(save_path + name + 'model.h5')



def CIDDS_sample(save_path, log_path, X_train, Y_train, name):
    """

    Args:
        save_path:
        log_path:
        X_train:
        Y_train:
        name:

    Returns:

    """
    save_log_file(log_path)

    ### 加上了Src、Dst特征
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '使用xgboost中的sample_weight参数进行样本平衡')

    model = mechine_learning.do_xgboost_blance_sample(X_train,Y_train)

    # create HDF5 file
    model.save(save_path + name + 'model.h5')


def CIDDS_scale(save_path, log_path, X_train, Y_train, name):
    """

    Args:
        save_path:
        log_path:
        X_train:
        Y_train:
        name:

    Returns:

    """
    save_log_file(log_path)

    ### 加上了Src、Dst特征
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '使用xgboost中的sample_weight参数进行样本平衡')

    model = mechine_learning.do_xgboost_blance_scale(X_train, Y_train)
    # create HDF5 file
    model.save(save_path + name + 'model.h5')


def Dnn(save_path, log_path, X_train, Y_train, name, shape):
    """

    Args:
        save_path:
        log_path:
        X_train:
        Y_train:
        name:

    Returns:

    """
    save_log_file(log_path)

    logging.info('用dnn完成实验分类,'
                 '使用sample_weight参数进行样本平衡')

    model = dnn.do_dnn_1d(X_train, Y_train, Input_shape=shape)

    # create HDF5 file
    model.save(save_path + name + 'model.h5')


def Dnn_sample(save_path, log_path, X_train, Y_train, name, shape):
    """

    Args:
        save_path:
        log_path:
        X_train:
        Y_train:
        name:

    Returns:

    """
    save_log_file(log_path)

    logging.info('用dnn完成实验分类,'
                 '使用sample_weight参数进行样本平衡')

    model = dnn.do_dnn_1d_sample(X_train, Y_train, Input_shape=shape)

    # create HDF5 file
    model.save(save_path + name + 'model.h5')

def TDR_sample(save_path, log_path, X_train, Y_train, name):
    """

    Args:
        save_path:
        log_path:
        X_train:
        Y_train:
        name:

    Returns:

    """
    save_log_file(log_path)

    ### 加上了Src、Dst特征
    logging.info('终端测试, 用xgboost完成实验分类,'
                 '使用xgboost中的sample_weight参数进行样本平衡')

    model = mechine_learning.do_xgboost_blance_sample(X_train,Y_train)

    # create HDF5 file
    model.save(save_path + name + 'model.h5')
