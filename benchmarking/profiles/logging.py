import logging


def get_logger(name=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(level)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(name)s - %(message)s'))
        logger.addHandler(sh)
    return logger
