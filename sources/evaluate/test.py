from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sources.utils.calculation_metrics import do_metrics
from sources.utils import confusion_matrixs
import numpy as np
from sources.utils.logfile import save_log_file
import logging
def CIDDS_test(X_test, Y_test, model_path, log_path,  name):
    """

    Args:
        X_test:
        Y_test:
        save_path:
        name:

    Returns:

    """
    save_log_file(log_path)
    model = load_model(model_path + name + 'model.h5')
    Y_test = to_categorical(Y_test, num_classes=5)
    Y_pred = model.predict(X_test)
    do_metrics(Y_test, Y_pred)
    attack_types = ['normal', 'attacker', 'victim', 'suspicious', 'unknown']
    confusion_matrixs.plot_confusion_matrix(np.array(metrics.confusion_matrix(Y_test, Y_pred)),
                                            classes=attack_types, normalize=True,
                                            title='xgb Normalized confusion matrix')


def dnn_test(X_test, Y_test, model_path, log_path, name):
    """

    Args:
        X_test:
        Y_test:
        save_path:
        name:

    Returns:

    """
    save_log_file(log_path)
    model = load_model(model_path + name + 'model.h5')
    Y_test = to_categorical(Y_test, num_classes=5)
    Y_pred = model.predict(X_test)
    do_metrics(Y_test, Y_pred)
    attack_types = ['normal', 'attacker', 'victim', 'suspicious', 'unknown']
    confusion_matrixs.plot_confusion_matrix(np.array(metrics.confusion_matrix(Y_test, Y_pred)),
                                            classes=attack_types, normalize=True,
                                            title='dnn Normalized confusion matrix')


def trd_test(X_test, Y_test, model_path, log_path, name):
    """

    Args:
        X_test:
        Y_test:
        save_path:
        name:

    Returns:

    """

    save_log_file(log_path)
    model = load_model(model_path + name + 'model.h5')
    Y_test = to_categorical(Y_test, num_classes=5)
    Y_pred = model.predict(X_test)
    do_metrics(Y_test, Y_pred)
    attack_types = ['normal', 'attacker', 'victim', 'suspicious', 'unknown']
    confusion_matrixs.plot_confusion_matrix(np.array(metrics.confusion_matrix(Y_test, Y_pred)),
                                            classes=attack_types, normalize=True,
                                            title='dnn Normalized confusion matrix')