from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import mechine_learning

import logging

if __name__=="__main__":

    log_path = '/home/liyulian/code/CIDDS/repositories'
    save_log_file(log_path)


    ### CIDDS-001
    path = '/home/liyulian/code/CIDDS/sources/utils/data_features_001.csv'

    ### 无样本平衡
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '无样本平衡, 设置了划分样本的random_state=1, 并且保存模型')

    X_train, X_test, Y_train, Y_test = get_features_FPPB(path)
    Y_pred, model = mechine_learning.do_xgboost(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)
    import pickle  # pickle模块

    # 保存Model(注:save文件夹要预先建立，否则会报错)
    with open('repositories/model_001.pickle', 'wb') as f:
        pickle.dump(model, f)

    # ### 无样本平衡
    # logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
    #              '无样本平衡')
    #
    # X_train, X_test, Y_train, Y_test = get_features(path)
    # Y_pred = mechine_learning.do_xgboost(X_train, X_test, Y_train, Y_test)
    # do_metrics(Y_test, Y_pred)

    """
    ### scale_pos_weight
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '使用xgboost中的scale_pos_weight参数进行样本平衡，scale_pos_weight=99')

    X_train, X_test, Y_train, Y_test = get_features_FPPB(path)
    Y_pred = mechine_learning.do_xgboost_blance_scale(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)

    ### sample_weight
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '使用xgboost中的sample_weight参数进行样本平衡')

    X_train, X_test, Y_train, Y_test = get_features_FPPB(path)
    Y_pred = mechine_learning.do_xgboost_blance_sample(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)
    """
    """
    ### 加上了Src、Dst特征
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '使用xgboost中的sample_weight参数进行样本平衡')

    X_train, X_test, Y_train, Y_test = get_features(path)
    Y_pred = mechine_learning.do_xgboost_blance_sample(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)
    
    ### 加上了Src、Dst特征
    logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                 '使用xgboost中的scale_pos_weight参数进行样本平衡，scale_pos_weight=99')

    X_train, X_test, Y_train, Y_test = get_features(path)
    Y_pred = mechine_learning.do_xgboost_blance_scale(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)
    """