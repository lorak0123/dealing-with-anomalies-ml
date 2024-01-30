import logging
import os


def exception_logger(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(e)
            if os.environ.get('DEBUG'):
                raise e
    return wrapper