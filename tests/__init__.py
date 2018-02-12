from __future__ import print_function

from os import getenv
import sys
from unittest import TestCase, skip, main

from progressivis import log_level, logging, Scheduler, BaseScheduler

import numpy as np

_ = skip # shut-up pylint



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
            self.log(int(level))
        else:
            self.log()

    def scheduler(self):
        if getenv("NOTHREAD"):
            if not self._output:
                print('[Using non-threaded scheduler]', end=' ', file=sys.stderr)
                self._output = True
            return BaseScheduler()
        return Scheduler()

    @staticmethod
    def log(level=logging.ERROR, package='progressivis'):
        log_level(level, package=package)

    @staticmethod
    def main():
        main()
