import logging
import traceback
import multiprocessing
import threading
import os

def get_logger(logging_file, enable_multiprocess, showing_stdout_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    file_ch = logging.FileHandler(logging_file)
    file_ch.setLevel(logging.DEBUG)
    
    if enable_multiprocess:
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] at [process_id: %(process)d] %(filename)s,%(lineno)d: %(message)s', 
                                            datefmt='%Y-%m-%d(%a)%H:%M:%S')
    else:
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s', 
                                            datefmt='%Y-%m-%d(%a)%H:%M:%S')
    file_ch.setFormatter(file_formatter)
    logger.addHandler(file_ch)

    #将大于或等于INFO级别的日志信息输出到StreamHandler(默认为标准错误)
    console = logging.StreamHandler()
    console.setLevel(showing_stdout_level) 
    formatter = logging.Formatter('[%(levelname)-8s] %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

'''
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result
'''