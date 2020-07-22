from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.utils.features import get_features
from sources.models import mechine_learning

import logging

if __name__=="__main__":

    log_path = '/home/liyulian/code/CIDDS/repositories'
    save_log_file(log_path)
    path = '/home/liyulian/code/CIDDS/sources/utils/data_features_002.csv'
    logging.info('使用CIDDS-002数据，该数据只有3类，且使用了Flags、attackType特征，这两类特征都是使用了one-hot方式，'
                 '没有用Proto、Src Pt和Dst Pt三个特征,最后用xgboost完成实验分类')

    X_train, X_test, Y_train, Y_test = get_features(path)
    Y_pred = mechine_learning.do_xgboost(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)