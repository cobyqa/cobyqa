import logging


def get_logger(name=None, level=logging.DEBUG):
    """
    Get a logger with the given name.

    Parameters
    ----------
    name : str
        Name of the logger. If None, the root logger is returned.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Logger with the given name.
    """
    logger = logging.getLogger(name)

    # Multiple calls to get_logger with the same name will return a reference
    # to the same logger. We do not recreate handlers that already exist.
    if len(logger.handlers) == 0:
        logger.setLevel(level)
        fmt = logging.Formatter('[%(asctime)s] %(levelname)-8s - %(message)s')

        # Attach a console handler (thread-safe).
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
