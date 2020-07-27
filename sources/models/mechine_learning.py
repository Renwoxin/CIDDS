# -*- coding: utf-8 -*-

"""
    This is a collection of machine learning algorithm codes for sequences

"""

import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import pickle

def do_xgboost(X_train, Y_train, model_path, name):
    """

    Args:
        X_train: DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        Y_train: list, the labels for training
        model_path: the save path of model
        name: the name of model

    Returns: None

    """
    xgb_model = xgb.XGBClassifier().fit(X_train, Y_train)
    with open(model_path+name+'model.pickle', 'wb') as f:
        pickle.dump(xgb_model, f)


def do_xgboost_blance_scale(X_train, X_test, Y_train, Y_test):
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
    Y_pred = xgb_model.predict(X_test)
    return Y_pred

def do_xgboost_blance_sample(X_train, X_test, Y_train, Y_test):
    """

    Args:
        X_train:DataFrame, shape(x_index*(1-test_size), feature_number(feature_map)), the features for training
        X_test: DataFrame, shape(x_index*test_size, feature_number(feature_map)), the features for testing
        Y_train: list, the labels for training
        Y_test: list, the labels for testing

    Returns:
        Y_pred: list, the Predicted value of the model

    """
    xgb_model = xgb.XGBClassifier().fit(X_train, Y_train,sample_weight=compute_sample_weight("balanced", Y_train))
    Y_pred = xgb_model.predict(X_test)
    return Y_pred
