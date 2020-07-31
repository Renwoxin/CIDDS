from sources.utils.logfile import save_log_file
from sources.models import machine_learnings
from sources.models import dnn
import pandas as pd
import numpy as np

import logging
import configparser
config = configparser.ConfigParser()
config.read('/home/liyulian/code/CIDDS/unittests/config.ini')

class xgb_train:
    def __init__(self):
        self.model_path = config['path', 'xgb_model_path']
        self.log_path = config['path', 'xgb_log_path']
        self.traindata_path = config['path', 'xgb_traindata_path']

        self.X_train = pd.read_csv(config['path', 'xgb_traindata_path'] + 'X_train.csv')
        self.Y_train = np.load(config['path', 'xgb_log_path'] + 'Y_train.npy')

    def xgb_none(self, name='none_xgb'):
        """

        Args:
            save_path:
            log_path:
            X_train:
            Y_train:
            name:

        Returns:

        """
        save_log_file(self.log_path)

        ### 无样本平衡
        logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                     '无样本平衡, 设置了划分样本的random_state=1, 并且保存模型')

        model = machine_learnings.do_xgboost(self.X_train,self.Y_train)

        # create HDF5 file
        model.save(self.model_path + name + 'model.h5')

    def xgb_sample(self, name='sample_xgb'):
        """

        Args:
            save_path:
            log_path:
            X_train:
            Y_train:
            name:

        Returns:

        """
        save_log_file(self.log_path)

        ### 加上了Src、Dst特征
        logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                     '使用xgboost中的sample_weight参数进行样本平衡')

        model = machine_learnings.do_xgboost_blance_sample(self.X_train, self.Y_train)

        # create HDF5 file
        model.save(self.model_path + name + 'model.h5')

    def xgb_scale(self, name='scale_xgb'):
        """

        Args:
            save_path:
            log_path:
            X_train:
            Y_train:
            name:

        Returns:

        """
        save_log_file(self.log_path)

        ### 加上了Src、Dst特征
        logging.info('使用CIDDS-001数据, 用xgboost完成实验分类,'
                     '使用xgboost中的sample_weight参数进行样本平衡')

        model = machine_learnings.do_xgboost_blance_scale(self.X_train, self.Y_train)
        # create HDF5 file
        model.save(self.model_path + name + 'model.h5')


class dnn_train:
    def __init__(self):
        self.model_path = config['path', 'xgb_model_path']
        self.log_path = config['path', 'xgb_log_path']
        self.traindata_path = config['path', 'xgb_traindata_path']
        self.result_path = config['path', 'dnn_result_path']
        self.X_train = pd.read_csv(config['path', 'xgb_traindata_path'] + 'X_train.csv')
        self.Y_train = np.load(config['path', 'xgb_log_path'] + 'Y_train.npy')
        self.shape = config['parameter', 'input_shape']

    def Dnn(self, name='dnn'):
        """

        Args:
            save_path:
            log_path:
            X_train:
            Y_train:
            name:

        Returns:

        """
        save_log_file(self.log_path)

        logging.info('用dnn完成实验分类,'
                     '使用sample_weight参数进行样本平衡')

        model = dnn.do_dnn_1d(self.X_train, self.Y_train, self.result_path, Input_shape=self.shape)

        # create HDF5 file
        model.save(self.model_path + name + 'model.h5')

    def Dnn_sample(self, name='sample_Dnn'):
        """

        Args:
            save_path:
            log_path:
            X_train:
            Y_train:
            name:

        Returns:

        """
        save_log_file(self.log_path)

        logging.info('用dnn完成实验分类,'
                     '使用sample_weight参数进行样本平衡')

        model = dnn.do_dnn_1d_sample(self.X_train, self.Y_train, self.result_path, Input_shape=self.shape)

        # create HDF5 file
        model.save(self.model_path + name + 'model.h5')

class trd_train:
    def __init__(self):
        self.model_path = config['path', 'xgb_model_path']
        self.log_path = config['path', 'xgb_log_path']
        self.traindata_path = config['path', 'xgb_traindata_path']

        self.X_train = pd.read_csv(config['path', 'xgb_traindata_path'] + 'X_train.csv')
        self.Y_train = np.load(config['path', 'xgb_log_path'] + 'Y_train.npy')

    def TDR_sample(self, name='sample_trd'):
        """

        Args:
            save_path:
            log_path:
            X_train:
            Y_train:
            name:

        Returns:

        """
        save_log_file(self.log_path)

        ### 加上了Src、Dst特征
        logging.info('终端测试, 用xgboost完成实验分类,'
                     '使用xgboost中的sample_weight参数进行样本平衡')

        model = machine_learnings.do_xgboost_blance_sample(self.X_train, self.Y_train)

        # create HDF5 file
        model.save(self.model_path + name + 'model.h5')
