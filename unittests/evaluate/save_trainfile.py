import unittest
from unittests.evaluate.test_my_train import Test_Mydnn_train


if __name__=='__main__':
    s = unittest.TestSuite()
    s.addTests([Test_Mydnn_train("test_sample_dnn")])
    fs = open("train_run_report.txt", "w")
    run = unittest.TextTestRunner(fs)
    run.run(s)