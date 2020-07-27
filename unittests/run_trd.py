from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import mechine_learning

import logging

if __name__ == "__main__":
    log_path = '/home/liyulian/code/CIDDS/repositories'
    save_log_file(log_path)

    ### CIDDS-002
    path = '/home/liyulian/data/data_features.csv'

    ### 加上了Src、Dst特征
    logging.info('使用CIDDS-002数据,'
                 '用xgboost完成实验分类,'
                 '终端测试')

    X_train, X_test, Y_train, Y_test = get_features(path)
    Y_pred = mechine_learning.do_xgboost_blance_sample(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)