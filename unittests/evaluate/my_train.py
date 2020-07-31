from sources.evaluate.train import *
import logging

class myclass_train:
    def __init__(self):
        pass
    def none_xgb(self):
        logging.info('Test case none_xgb')
        xgb = xgb_train()
        xgb.xgb_none()
        return True

    def scale_xgb(self):
        logging.info('Test case scale_xgb')
        xgb = xgb_train()
        xgb.xgb_scale()
        return True

    def sample_xgb(self):
        logging.info('Test case sample_xgb')
        xgb = xgb_train()
        xgb.xgb_sample()
        return True

    def sample_TDR(self):
        logging.info('Test case sample_TDR')
        xgb = trd_train()
        xgb.TDR_sample()
        return True

class mydnn_train:
    def __init__(self):
        pass

    def none_dnn(self):
        logging.info('Test case none_dnn')
        dnn = dnn_train()
        dnn.Dnn()
        return True

    def sample_dnn(self):
        logging.info('Test case sample_dnn')
        dnn = dnn_train()
        dnn.Dnn_sample()
        return True

