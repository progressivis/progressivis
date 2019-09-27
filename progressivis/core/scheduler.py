"""Multi-Thread Scheduler, meant to run in its own thread."""
from __future__ import absolute_import, division, print_function

import logging
from .synchronized import synchronized
from .scheduler_base import BaseScheduler
from .utils import ProgressiveError, Thread, RLock

logger = logging.getLogger(__name__)


class Scheduler(BaseScheduler):
    """
    Main scheduler class.

    Manage the execution of the progressive workflow in its own thread.
    """
    def __init__(self):
        super(Scheduler, self).__init__()
        self.thread = None
        self.thread_name = "Progressive Scheduler"

    def create_lock(self):
        return RLock()

    def join(self):
        with self.lock:
            if self.thread is None:
                return
        self.thread.join()

    @staticmethod
    def set_default():
        "Set the default scheduler."
        if not isinstance(BaseScheduler.default, Scheduler):
            BaseScheduler.default = Scheduler()

    @synchronized
    def validate(self):
        return super(Scheduler, self).validate()

    @synchronized
    def invalidate(self):
        super(Scheduler, self).invalidate()

    def _before_run(self):
        logger.debug("Before run %d", self._run_number)

    def start(self, tick_proc=None, idle_proc=None):
        with self.lock:
            if self.thread is not None:
                raise ProgressiveError('Trying to start scheduler thread'
                                       ' inside scheduler thread')
            self.thread = Thread(target=self.run, name=self.thread_name)
            if tick_proc:
                assert callable(tick_proc)
                self._tick_procs = [tick_proc]
            else:
                self._tick_procs = []
            if idle_proc:
                assert callable(idle_proc)
                self._idle_procs = [idle_proc]
            else:
                self._idle_procs = []
            logger.info('starting thread')
        self.thread.start()

    def _after_run(self):
        logger.debug("After run %d", self._run_number)

    def done(self):
        self.thread = None


if BaseScheduler.default is None:
    BaseScheduler.default = Scheduler()
