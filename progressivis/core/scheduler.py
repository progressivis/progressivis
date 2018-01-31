"""Multi-Thread Scheduler, meant to run in its own thread."""
from __future__ import absolute_import, division, print_function

import sys
import io
from tempfile import mkstemp
from contextlib import contextmanager
import logging
from .synchronized import synchronized
from .scheduler_base import BaseScheduler
from .utils import ProgressiveError, Thread, RLock

logger = logging.getLogger(__name__)


class Scheduler(BaseScheduler):
    def __init__(self):
        super(Scheduler, self).__init__()
        self.thread = None
        self._thread_parent = None
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
        if not isinstance(BaseScheduler.default, Scheduler):
            BaseScheduler.default = Scheduler()

    @staticmethod
    def log_level(level=logging.DEBUG, package='progressivis'):
        logging.getLogger('progressivis').addHandler(logging.NullHandler())
        filedesc, _ = mkstemp(prefix='progressive', suffix='.log')
        stream = io.FileIO(filedesc, mode='w')
        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        l = logging.getLogger(package)
        l.addHandler(ch)
        l.setLevel(level)
        l.propagate = False

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
            if hasattr(sys.stdout, 'thread_parent'):
                # capture notebook context
                self._thread_parent = sys.stdout.thread_parent
            else:
                self._thread_parent = None
            self._tick_proc = tick_proc
            self._idle_proc = idle_proc
            logger.debug('starting thread')
        self.thread.start()

    def _after_run(self):
        logger.debug("After run %d", self._run_number)

    def stop(self):
        super(Scheduler, self).stop()

    def done(self):
        self.thread = None
        self._thread_parent = None

    @property
    def thread_parent(self):
        return self._thread_parent

    @contextmanager
    def stdout_parent(self):
        if self._thread_parent:
            saved_parent = sys.stdout.parent_header
            with self.lock:
                sys.stdout.parent_header = self._thread_parent
                try:
                    yield
                finally:
                    sys.stdout.flush()
                    sys.stdout.parent_header = saved_parent
        else:
            yield


if BaseScheduler.default is None:
    BaseScheduler.default = Scheduler()
