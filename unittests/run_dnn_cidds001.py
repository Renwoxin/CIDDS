from sources.utils.logfile import save_log_file
from sources.utils.calculation_metrics import do_metrics
from sources.preprocess.features import *
from sources.models import dnn
from tensorflow.keras.models import load_model
from sources.utils import confusion_matrixs
from sklearn import metrics

import logging

if __name__ == "__main__":
    log_path = '/home/liyulian/code/CIDDS/repositories/dnn'
    save_log_file(log_path)

    ### CIDDS-001
    path = '/home/liyulian/code/CIDDS/sources/utils/data_features_001.csv'
    save_path = '/home/liyulian/code/CIDDS/repositories/dnn/model'
    name = 'hash'
    ### non-sample balance
    logging.info('Using CIDDS-001 data, '
                 'using dnn to complete the experimental classification without sample balance, '
                 'training and testing by saving the model and then loading the model')

    X_train, X_test, Y_train, Y_test = get_features(path)
    dnn.do_dnn_1d(X_train, Y_train, save_path=save_path, name=name, Input_shape=44)
    model = load_model(save_path+name+'model.h5')
    Y_pred = model.predict(Y_test)
    do_metrics(Y_test, Y_pred)
    attack_types = ['normal', 'attacker', 'victim', 'suspicious', 'unknown']
    confusion_matrixs.plot_confusion_matrix(np.array(metrics.confusion_matrix(Y_test, Y_pred)),
                                            classes=attack_types, normalize=True,
                                            title='xgb Normalized confusion matrix')
