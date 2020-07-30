######定义测试类1：test_Myclass1.py

import unittest
from unittests.evaluate.my_test import myclass_test
from sources.preprocess.features import *
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

path = '/home/liyulian/code/CIDDS/sources/utils/data_features_001.csv'
save_path = '/home/liyulian/code/CIDDS/repositories/xgb_model/'
log_path = '/home/liyulian/code/CIDDS/repositories/Logs/'
path_test = '/home/liyulian/code/CIDDS/repositories/TDR/data/data_features_test_unique_1000_35.csv'

X, Y = get_features(path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_test_, Y_test_ = get_feature(path_test)
class Test_Myclass_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.info("在所有的测试用例执行之前，只执行一次============")

    @classmethod
    def tearDownClass(cls):
        logging.info("在所有的测试用例执行之后，只执行一次============")


    def test_tests_CIDDS(self):
        logging.info("result")
        name = 'none_CIDDS'

        cal = myclass_test(X_test, Y_test, save_path, log_path, name)
        result = cal.CIDDS_tests()
        self.assertEqual(True, result)

    def test_trd(self):
        logging.info("result")
        name = 'none_trd'
        cal = myclass_test(X_test, Y_test, save_path, log_path, name)
        result = cal.trd_tests()
        self.assertEqual(True, result)




