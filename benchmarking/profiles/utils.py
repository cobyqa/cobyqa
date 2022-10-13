import logging


def get_logger(name=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
