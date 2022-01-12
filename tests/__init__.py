from __future__ import annotations

from os import getenv
import gc
import sys
from unittest import TestCase, main
from unittest import skip as skip
from unittest import skipIf as skipIf
import logging

from progressivis import Scheduler, log_level
from progressivis.storage import init_temp_dir_if, cleanup_temp_dir
import numpy as np

from typing import Any, Type, Optional

_ = skip  # shut-up pylint
__ = skipIf


class ProgressiveTest(TestCase):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    def __init__(self, *args: Any) -> None:
        super(ProgressiveTest, self).__init__(*args)
        self._output: bool = False
        self._scheduler: Optional[Scheduler] = None
        self._temp_dir_flag: bool = False
        level: Any = getenv("LOGLEVEL")
        if level in ProgressiveTest.levels:
            level = ProgressiveTest.levels[level]
        if level:
            print(f"Logger level {level} for {self}", file=sys.stderr)
            self.log(int(level))

    @staticmethod
    def terse(x: Any) -> None:
        _ = x
        print(".", end="", file=sys.stderr)

    @staticmethod
    async def _stop(scheduler: Scheduler, run_number: int) -> None:
        await scheduler.stop()

    def setUp(self) -> None:
        np.random.seed(42)

    def tearDown(self) -> None:
        # print('Logger level for %s back to ERROR' % self, file=sys.stderr)
        # self.log()
        gc.collect()
        logger = logging.getLogger()
        logger.setLevel(logging.NOTSET)
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

    @classmethod
    def cleanup(self) -> None:
        cleanup_temp_dir()

    @classmethod
    def setUpClass(cls: Type[ProgressiveTest]) -> None:
        cleanup_temp_dir()
        init_temp_dir_if()

    @classmethod
    def tearDownClass(cls: Type[ProgressiveTest]) -> None:
        cleanup_temp_dir()

    def scheduler(self, clean: bool = False) -> Scheduler:
        if self._scheduler is None or clean:
            self._scheduler = Scheduler()
        return self._scheduler

    @staticmethod
    def log(level: int = logging.NOTSET, package: str = "progressivis") -> None:
        log_level(level, package=package)

    @staticmethod
    def main() -> None:
        main()
