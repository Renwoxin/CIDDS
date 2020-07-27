from sources.preprocess.features import *
from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.models import mechine_learning

import logging

if __name__=='__main__':
    log_path = '/home/liyulian/code/CIDDS/repositories'
    save_log_file(log_path)

    ### CIDDS-001
    path = '/home/liyulian/code/CIDDS/sources/utils/data_features_001_E.csv'

    ### scale_pos_weight
    logging.info('使用CIDDS-001_E数据, 用xgboost完成实验分类,'
                 '使用xgboost中的sample_weight参数进行样本平衡.')

    X_train, X_test, Y_train, Y_test = get_features_hash(path)
    Y_pred = mechine_learning.do_xgboost_blance_sample(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)