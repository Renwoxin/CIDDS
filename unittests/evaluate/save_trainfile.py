import unittest
from unittests.evaluate.test_my_train import Test_Myclass_train


if __name__=='__main__':
    s = unittest.TestSuite()
    s.addTest(Test_Myclass_train("test_none_CIDDS"))
    s.addTests([Test_Myclass_train("test_none_CIDDS"), Test_Myclass_train("test_scale_CIDDS"), Test_Myclass_train("test_sample_CIDDS")])
    fs = open("train_run_report.txt", "w")
    run = unittest.TextTestRunner(fs)
    run.run(s)