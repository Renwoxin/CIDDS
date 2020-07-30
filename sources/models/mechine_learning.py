# -*- coding: utf-8 -*-

"""
    This is a collection of machine learning algorithm codes for sequences

"""
from sources.utils.calculation_metrics import do_metrics
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def do_xgboost(X_train, Y_train):
    """

    Args:
        X_train: DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        Y_train: list, the labels for training
        model_path: the save path of model
        name: the name of model

    Returns: None

    """
    xgb_model = xgb.XGBClassifier().fit(X_train, Y_train)

    return xgb_model


def do_xgboost_blance_scale(X_train, Y_train):
    """

    Args:
        X_train:DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        X_test: DataFrame, shape(x_index*test_size, feature_number(feature_map)), the features for testing
        Y_train: list, the labels for training
        Y_test: list, the labels for testing

    Returns:
        Y_pred: list, the Predicted value of the model

    """
    xgb_model = xgb.XGBClassifier(scale_pos_weight=99).fit(X_train, Y_train)
    return xgb_model

def do_xgboost_blance_sample(X, Y):
    """

    Args:
        X_train:DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        X_test: DataFrame, shape(x_index*test_size, feature_number(feature_map)), the features for testing
        Y_train: list, the labels for training
        Y_test: list, the labels for testing

    Returns:
        Y_pred: list, the Predicted value of the model

    """
    logging.info('训练结果输出')
    X_train, X_vld, Y_train, Y_vld = train_test_split(X, Y, test_size=0.2, random_state=1)
    xgb_model = xgb.XGBClassifier().fit(X_train, Y_train,sample_weight=compute_sample_weight("balanced", Y_train))
    Y_pred = xgb_model.predict(X_vld)
    do_metrics(Y_vld, Y_pred)
    return xgb_model
