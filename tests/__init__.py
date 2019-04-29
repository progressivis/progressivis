from __future__ import print_function

from os import getenv
import sys
from unittest import TestCase, skip, skipIf, main

from progressivis import (log_level, logging,
                          Scheduler, BaseScheduler, 
                          Dataflow)
from progressivis.storage import Group, StorageEngine
from progressivis.storage.mmap import MMapGroup
import numpy as np

_ = skip # shut-up pylint
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
        self._schedulers = []
        self._dataflow = None

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
        sched = None
        if getenv("NOTHREAD"):
            if not self._output:
                print('[Using non-threaded scheduler]', end=' ', file=sys.stderr)
                self._output = True
            sched = BaseScheduler()
        else:
            sched = Scheduler()
        self._schedulers.append(sched)
        return sched

    def dataflow(self):
        if self._dataflow is None:
            self._dataflow = Dataflow()
        return self._dataflow

    @staticmethod
    def log(level=logging.ERROR, package='progressivis'):
        log_level(level, package=package)

    @staticmethod
    def main():
        main()
