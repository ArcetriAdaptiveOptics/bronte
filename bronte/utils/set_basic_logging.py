import importlib
import logging

def set_basic_logging(debug_class_name="ClassName"):

    importlib.reload(logging)
    FORMAT = '%(asctime)s:%(levelname)s:%(name)s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logger = logging.getLogger(debug_class_name)
    
    return logger