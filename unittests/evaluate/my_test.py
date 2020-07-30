from sources.evaluate.test import *
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class myclass_test:
    def __init__(self,  X_test, Y_test, model_path, log_path, name):
        self.X_test = X_test
        self.Y_test = Y_test
        self.save_path = model_path
        self.log_path = log_path
        self.name = name

    def CIDDS_tests(self):
        logging.info('Test case CIDDS_tests')
        CIDDS_test(self.X_test, self.Y_test, self.save_path, self.log_path, self.name)
        return True

    def dnn_tests(self):
        logging.info('Test case dnn_tests')
        dnn_test(self.X_test, self.Y_test, self.save_path, self.log_path, self.name)
        return True

    def trd_tests(self):
        logging.info('Test case dnn_tests')
        trd_test(self.X_test, self.Y_test, self.save_path, self.log_path, self.name)
        return True