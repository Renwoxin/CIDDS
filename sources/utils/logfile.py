import logging  # 引入logging模块
import time


def save_log_file(path):
    """

    Args:
        path: the save path for log file

    Returns: None

    """
    # the first step，create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log level master switch
    # the second step, create a handler，Used to write log files
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = path + '/Logs/'
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # Switch of log level output to file
    # The third step , defining the output format of the handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # The fourth step, adding the logger to the handler
    logger.addHandler(fh)