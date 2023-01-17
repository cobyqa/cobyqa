import logging
from io import StringIO


def get_logger(name=None, level=logging.DEBUG):
    logger = logging.getLogger(name)

    # Multiple calls to getLogger() with the same name will return a reference
    # to the same logger object. Therefore, we do not need to recreate the
    # handlers if they already exist.
    if len(logger.handlers) == 0:
        logger.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s - %(message)s")

        # Create a console handler (thread-safe).
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # TODO: Create a file handler. The FileHandler class is not thread-safe.
        #  We should have all the processes log to a SocketHandler, and have a
        #  separate process which implements a socket server which reads from
        #  the socket and logs to file.
    return logger


class NullIO(StringIO):

    def write(self, __s):
        pass
