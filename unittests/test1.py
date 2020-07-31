from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import machine_learnings
from sklearn.model_selection import ShuffleSplit
from sources.utils import plot_result
import logging

if __name__ == "__main__":
    log_path = '/home/liyulian/code/CIDDS/repositories'
    save_log_file(log_path)

    ### CIDDS-002
    path = '/home/liyulian/code/CIDDS/sources/utils/data_features_002.csv'


    ### scale_pos_weight
    logging.info('使用CIDDS-002数据, 用xgboost完成实验分类,'
                 '使用xgboost中的scale_pos_weight参数进行样本平衡，scale_pos_weight=99')
    # 图一
    title = r"Learning Curves (Naive Bayes)"

    X_train, X_test, Y_train, Y_test = get_features_FPPB(path)
    Y_pred, model= machine_learnings.do_xgboost(X_train, X_test, Y_train, Y_test)
    do_metrics(Y_test, Y_pred)

    x = np.concatenate([X_train, X_test])
    y = np.concatenate([Y_train,Y_test])

    cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=0)
    plot_result.plot_learning_curve(model, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

