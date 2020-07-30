from sources.evaluate.train import *
import logging

class myclass_train:
    def __init__(self, save_path, log_path, X_train, Y_train, name):
        self.save_path = save_path
        self.log_path = log_path
        self.X_train = X_train
        self.Y_train = Y_train
        self.name = name

    def none_CIDDS(self):
        logging.info('Test case none_CIDDS')
        CIDDS_none(self.save_path,self.log_path, self.X_train, self.Y_train, self.name)
        return True

    def scale_CIDDS(self):
        logging.info('Test case scale_CIDDS')
        CIDDS_scale(self.save_path, self.log_path, self.X_train, self.Y_train, self.name)
        return True

    def sample_CIDDS(self):
        logging.info('Test case sample_CIDDS')
        CIDDS_sample(self.save_path, self.log_path, self.X_train, self.Y_train, self.name)
        return True

    def sample_TDR(self):
        logging.info('Test case sample_TDR')
        TDR_sample(self.save_path, self.log_path, self.X_train, self.Y_train, self.name)
        return True

class mydnn_train:
    def __init__(self, save_path, log_path, X_train, Y_train, name, shape):
        self.save_path = save_path
        self.log_path = log_path
        self.X_train = X_train
        self.Y_train = Y_train
        self.name = name
        self.shape = shape

    def none_dnn(self):
        logging.info('Test case none_dnn')
        Dnn(self.save_path, self.log_path, self.X_train, self.Y_train, self.name, self.shape)
        return True

    def sample_dnn(self):
        logging.info('Test case sample_dnn')
        Dnn(self.save_path, self.log_path, self.X_train, self.Y_train, self.name, self.shape)
        return True

