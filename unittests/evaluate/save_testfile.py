import unittest
from unittests.evaluate.test_my_test import Test_Myclass_test


if __name__=='__main__':
    s = unittest.TestSuite()
    s.addTest(Test_Myclass_test("test_tests_CIDDS"))
    fs = open("test_run_report.txt", "w")
    run = unittest.TextTestRunner(fs)
    run.run(s)