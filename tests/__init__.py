from __future__ import print_function

from os import getenv
import sys
from unittest import TestCase, skip, skipIf, main

from progressivis import (log_level, logging,
                          Scheduler, BaseScheduler)
from progressivis.storage import StorageEngine
import numpy as np

_ = skip  # shut-up pylint
_ = skipIf


class ProgressiveTest(TestCase):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET
    levels = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET
    }

    def __init__(self, *args):
        super(ProgressiveTest, self).__init__(*args)
        self._output = False
        self._scheduler = None

    @staticmethod
    def terse(x):
        _ = x
        print('.', end='', file=sys.stderr)

    def setUp(self):
        np.random.seed(42)
        level = getenv("LOGLEVEL")
        if level in self.levels:
            level = self.levels[level]
        if level:
            print('Logger level: ', level, file=sys.stderr)
            self.log(int(level))
        else:
            self.log()

    @classmethod
    def cleanup(self):
        if StorageEngine.default == 'mmap':
            root = StorageEngine.engines()['mmap']
            if not root.has_files():
                return
            root.close_all()
            root.delete_children()
            root.dict = {}

    @classmethod
    def setUpClass(cls):
        cls.cleanup()

    @classmethod
    def tearDownClass(cls):
        cls.cleanup()

    def scheduler(self):
        if self._scheduler is None:
            if getenv("NOTHREAD"):
                if not self._output:
                    print('[Using non-threaded scheduler]', end=' ', file=sys.stderr)
                    self._output = True
                self._scheduler = BaseScheduler()
            else:
                self._scheduler = Scheduler()
        return self._scheduler

    @staticmethod
    def log(level=logging.ERROR, package='progressivis'):
        log_level(level, package=package)

    @staticmethod
    def main():
        main()
