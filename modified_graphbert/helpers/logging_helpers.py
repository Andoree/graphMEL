import logging


class Writer(object):
    def __init__(self, level):
        self.level = level
        self.log = []

    def write(self, data):
        self.log.append(data)

    def flush(self):
        if self.log:
            self.level(" ".join(self.log))


class ExitOnExceptionHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            raise SystemExit(-1)
