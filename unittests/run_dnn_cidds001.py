from sources.utils.logfile import save_log_file
from sources.preprocess.features import *
from sources.models import dnn
import numpy as np

import logging

if __name__ == "__main__":
    log_path = '/home/liyulian/code/CIDDS/repositories/dnn'
    save_log_file(log_path)

    ### CIDDS-001
    path = '/home/liyulian/code/CIDDS/sources/utils/data_features_001.csv'
    save_model_path = '/home/liyulian/code/CIDDS/repositories/dnn/model/'
    save_data_path = '/home/liyulian/code/CIDDS/repositories/dnn/data/'
    name = 'hash'
    ### non-sample balance
    # logging.info('Using CIDDS-001 data, '
    #              'using dnn to complete the experimental classification without sample balance, '
    #              'training and testing by saving the model and then loading the model')
    #
    # X, Y = get_features(path)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # X_train.to_csv(save_data_path + 'x_traindata.csv')
    # np.save(save_data_path + 'y_traindata.npy', Y_train)
    # X_test.to_csv(save_data_path + 'x_testdata.csv')
    # np.save(save_data_path + 'y_testdata.npy', Y_test)

    X_train = pd.read_csv(save_data_path + 'x_traindata.csv', low_memory=False)
    Y_train = np.load(save_data_path + 'y_traindata.npy', )
    dnn.do_dnn_1d(X_train, Y_train, save_path=save_model_path, name=name, Input_shape=44)
    # model = load_model(save_path+name+'model.h5')
    # Y_test = to_categorical(Y_test, num_classes=5)
    # Y_pred = model.predict(X_test)
    # do_metrics(Y_test, Y_pred)
    # attack_types = ['normal', 'attacker', 'victim', 'suspicious', 'unknown']
    # confusion_matrixs.plot_confusion_matrix(np.array(metrics.confusion_matrix(Y_test, Y_pred)),
    #                                         classes=attack_types, normalize=True,
    #                                         title='xgb Normalized confusion matrix')
