from os import getenv
import sys
from unittest import TestCase, skip, skipIf, main

from progressivis import (log_level, logging, Scheduler)
from progressivis.storage import init_temp_dir_if, cleanup_temp_dir
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
        self._temp_dir_flag = False
        level = getenv("LOGLEVEL")
        if level in self.levels:
            level = self.levels[level]
        if level:
            print('Logger level for %s: ' % self, level, file=sys.stderr)
            self.log(int(level))

    @staticmethod
    def terse(x):
        _ = x
        print('.', end='', file=sys.stderr)

    def setUp(self):
        np.random.seed(42)

    def tearDown(self):
        # print('Logger level for %s back to ERROR' % self, file=sys.stderr)
        self.log()

    @classmethod
    def cleanup(self):
        cleanup_temp_dir()

    @classmethod
    def setUpClass(cls):
        cleanup_temp_dir()
        init_temp_dir_if()

    @classmethod
    def tearDownClass(cls):
        cleanup_temp_dir()

    def scheduler(self, clean=False):
        if self._scheduler is None or clean:
            self._scheduler = Scheduler()
        return self._scheduler

    @staticmethod
    def log(level=logging.ERROR, package='progressivis'):
        log_level(level, package=package)

    @staticmethod
    def main():
        main()
