import logging
from sklearn import metrics

def do_metrics(Y_test,Y_pred):
    """

    Args:
        Y_test: list, the labels for testing
        Y_pred: list, the labels for prediction

    Returns:None

    """
    logging.info("metrics.accuracy_score:")
    logging.info(metrics.accuracy_score(Y_test, Y_pred))
    logging.info("metrics.confusion_matrix:")
    logging.info(metrics.confusion_matrix(Y_test, Y_pred))
    logging.info("metrics.precision_score:")
    logging.info(metrics.precision_score(Y_test, Y_pred, average='macro'))
    logging.info(metrics.precision_score(Y_test, Y_pred, average='micro'))
    logging.info(metrics.precision_score(Y_test, Y_pred, average='weighted'))

    logging.info("metrics.recall_score:")
    logging.info(metrics.recall_score(Y_test, Y_pred, average='macro'))
    logging.info(metrics.recall_score(Y_test, Y_pred, average='micro'))
    logging.info(metrics.recall_score(Y_test, Y_pred, average='weighted'))

    logging.info("metrics.f1_score:")
    logging.info(metrics.f1_score(Y_test,Y_pred, average='macro'))
    logging.info(metrics.f1_score(Y_test, Y_pred, average='micro'))
    logging.info(metrics.f1_score(Y_test, Y_pred, average='weighted'))
