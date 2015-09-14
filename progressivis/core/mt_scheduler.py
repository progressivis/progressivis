"""Multi-Thread Scheduler, meant to run in its own thread."""
from scheduler import Scheduler
import threading
import sys, io
from tempfile import mkstemp

from contextlib import contextmanager

import logging
logger = logging.getLogger(__name__)


class MTScheduler(Scheduler):
    def __init__(self):
        super(MTScheduler,self).__init__()
        self.lock = threading.RLock()
        self.thread = None
        self._thread_parent = None

    @staticmethod
    def set_default():
        if not isinstance(Scheduler.default, MTScheduler):
            Scheduler.default = MTScheduler()

    @staticmethod
    def log_level(level=logging.DEBUG, package='progressivis'):
        logging.getLogger('progressivis').addHandler(logging.NullHandler())
        fd, filename = mkstemp(prefix='progressive', suffix='.log')
        stream = io.FileIO(fd, mode='w')
        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        l=logging.getLogger(package)
        l.addHandler(ch)
        l.setLevel(level)
        l.propagate = False

    def collect_dependencies(self, only_required=False):
        with self.lock:
            return super(MTScheduler,self).collect_dependencies(only_required)

    def validate(self):
        with self.lock:
            return super(MTScheduler,self).validate()

    def invalidate(self):
        with self.lock:
            super(MTScheduler,self).invalidate()

    def _before_run(self):
        logger.debug("Before run %d" % self._run_number)


    def start(self, tick_proc=None):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, name="Progressive Scheduler")
            if hasattr(sys.stdout, 'thread_parent'):
                self._thread_parent = sys.stdout.thread_parent # capture notebook context
            else:
                self._thread_parent = None
            self._tick_proc = tick_proc
            self.thread.start()
            logger.debug('starting thread')
        else:
            raise ProgressiveError('Trying to start scheduler thread inside scheduler thread')

    def _after_run(self):
        #print "After run %d" % self._run_number
        logger.debug("After run %d" % self._run_number)

    def stop(self):
        super(MTScheduler,self).stop()

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

    def _add_module(self, module):
        with self.lock:
            super(MTScheduler,self)._add_module(module)

    def _remove_module(self, module):
        with self.lock:
            super(MTScheduler,self)._remove_module(module)
        
