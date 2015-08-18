"""Multi-Thread Scheduler, meant to run in its own thread."""
from scheduler import Scheduler
import threading
import sys

class MTScheduler(Scheduler):
    def __init__(self):
        super(MTScheduler,self).__init__()
        self.lock = threading.RLock()
        self.thread = None
        self.debug = False
        self._thread_parent = None

    @staticmethod
    def install():
        if not isinstance(Scheduler.default, MTScheduler):
            Scheduler.default = MTScheduler()

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
        if self.debug:
            print "Before run %d" % self._run_number


    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, name="Progressive Scheduler")
            if sys.stdout.hasattr('thread_parent'):
                self._thread_parent = sys.stdout.thread_parent # capture notebook context
            else:
                self._thread_parent = None
            self.thread.start()
        else:
            raise ProgressiveError('Trying to start scheduler thread inside scheduler thread')

    def _after_run(self):
        if self.debug:
            print "After run %d" % self._run_number

    def stop(self):
        super(MTScheduler,self).stop()

    def done(self):
        self.thread = None
        self._thread_parent = None

    @property
    def thread_parent(self):
        return self._thread_parent

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
        
