######定义测试类1：test_Myclass1.py

import unittest
from unittests.evaluate.my_train import myclass_train, mydnn_train
from sources.preprocess.features import *
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# del(X)
# del(Y)
# path_001 = '/home/liyulian/data/CIDDS/CIDDS-001/traffic'
# path_002 = '/home/liyulian/data/CIDDS/CIDDS-002/traffic'
# save_data_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/'
# data_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/data_features_test_10000_35.csv'
# dic1_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/Flags_dic.npy'
# dic2_path = '/home/liyulian/code/CIDDS/repositories/TDR/data/Proto_dic.npy'
# path_train = '/home/liyulian/code/CIDDS/repositories/TDR/data/data_features_train_unique.csv'
# log_path_ = '/home/liyulian/code/CIDDS/repositories/TDR'
# X_train_, Y_train_ = get_feature(path_train)


# class Test_Myclass_train(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         logging.info("在所有的测试用例执行之前，只执行一次============")
#
#     @classmethod
#     def tearDownClass(cls):
#         logging.info("在所有的测试用例执行之后，只执行一次============")
#
#
#     def test_none_CIDDS(self):
#         logging.info("result_none")
#         name = 'none_CIDDS'
#
#         cal = myclass_train(save_path, log_path, X_train, Y_train, name)
#         result = cal.none_CIDDS()
#         self.assertEqual(True,result)
#
#     def test_scale_CIDDS(self):
#         logging.info("result_sacle")
#         name = 'scale_CIDDS'
#         cal = myclass_train(save_path, log_path, X_train, Y_train, name)
#         result = cal.scale_CIDDS()
#         self.assertEqual(True,result)
#
#     def test_sample_CIDDS(self):
#         logging.info("result_sample")
#         name = 'sample_CIDDS'
#         cal = myclass_train(save_path, log_path, X_train, Y_train, name)
#         result = cal.sample_CIDDS()
#         self.assertEqual(True,result)
#
#     def test_sample_TRD(self):
#         logging.info("result_sample")
#         name = 'sample_TDR'
#         cal = myclass_train(save_data_path, log_path_, X_train_, Y_train_, name)
#         result = cal.sample_TDR()
#         self.assertEqual(True,result)


class Test_Mydnn_train(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.info("在所有的测试用例执行之前，只执行一次============")

    @classmethod
    def tearDownClass(cls):
        logging.info("在所有的测试用例执行之后，只执行一次============")


    def test_none_dnn(self):
        logging.info("result_none")
        name = 'none_dnn'
        cal = mydnn_train()
        result = cal.none_dnn()
        self.assertEqual(True,result)

    def test_sample_dnn(self):
        logging.info("result_none")
        name = 'sample_dnn'

        cal = mydnn_train()
        result = cal.sample_dnn()
        self.assertEqual(True, result)




