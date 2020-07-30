from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import mechine_learning

import logging

if __name__ == "__main__":
    log_path = '/home/liyulian/code/CIDDS/repositories/TDR'
    save_log_file(log_path)

    ### CIDDS-002
    path_train = '/home/liyulian/code/CIDDS/repositories/TDR/data/data_features_train_unique.csv'
    path_test = '/home/liyulian/code/CIDDS/repositories/TDR/data/data_features_test_unique_1000_35.csv'

    ### 加上了Src、Dst特征

    logging.info('使用所有的包含该主机的数据作为训练集，另外的200主机的数据作为测试集,'
                 '用xgboost完成实验分类,'
                 '终端测试1000_35')

    X_train, Y_train = get_feature(path_train)
    X_test, Y_test = get_feature(path_test)
    model = mechine_learning.do_xgboost_blance_sample(X_train, Y_train)
    Y_pred = model.predict(X_test)
    do_metrics(Y_test, Y_pred)