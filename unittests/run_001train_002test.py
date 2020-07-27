from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import mechine_learning
from sources.utils import confusion_matrixs
from sklearn import metrics

import pickle
import logging

if __name__=="__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = '/home/liyulian/code/CIDDS/repositories'
    save_log_file(log_path)

    ### CIDDS-001 data set as a training set, CIDDS-002 data set as a testing set
    path_1 = '/home/liyulian/code/CIDDS/sources/utils/data_features_001.csv'
    path_2 = '/home/liyulian/code/CIDDS/sources/utils/data_features_002.csv'
    model_path = '/home/liyulian/code/CIDDS/repositories/xgb_model'
    ### sample_weight
    logging.info('CIDDS-001 data set as a training set, CIDDS-002 data set as a testing set,'
                 'Use the "sample_weight" parameter in xgboost for sample balance')

    X_train, Y_train = get_features_train_test(path_1)

    mechine_learning.do_xgboost(X_train, Y_train, model_path, '001train_002test')

    X_test, Y_test = get_features_train_test(path_2)

    # read Model
    with open(model_path + '001train_002test' + 'model.pickle', 'rb') as f:
        xgb_model = pickle.load(f)

    # test the Model
    attack_types = ['normal', 'attacker', 'victim']

    Y_pred = xgb_model.predict(X_test)
    do_metrics(Y_test, Y_pred)
    confusion_matrixs.plot_confusion_matrix(np.array(metrics.confusion_matrix(Y_test,Y_pred)),
                                            classes=attack_types, normalize=True, title='xgb Normalized confusion matrix')