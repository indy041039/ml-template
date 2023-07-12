import logging
import io


def create_logger(name, logfile_path, is_stream=True, is_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear previous logger
    if (logger.hasHandlers()):
        logger.handlers.clear()

    if is_stream:
        handler = logging.StreamHandler()
        format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(format)
        logger.addHandler(handler)
    
    if is_file:
        fhandler = logging.FileHandler(logfile_path, 'w+')
        fformat = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(fformat)
        logger.addHandler(fhandler)
    
    return logger
