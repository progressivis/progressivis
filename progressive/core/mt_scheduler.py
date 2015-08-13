"""Multi-Thread Scheduler, meant to run in its own thread."""
from scheduler import Scheduler
import threading

class MTScheduler(Scheduler):
    def __init__(self):
        super(MTScheduler,self).__init__()
        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self.run, name="Progressive Scheduler")

    @staticmethod
    def install():
        if not isinstance(Scheduler.default, MTScheduler):
            Scheduler.default = MTScheduler()

    def collect_dependencies(self):
        with self.lock:
            super(MTScheduler,self).collect_dependencies()

    def validate(self):
        with self.lock:
            super(MTScheduler,self).validate()

    def invalidate(self):
        with self.lock:
            super(MTScheduler,self).invalidate()
    
    def run(self):
        if threading.current_thread()!=self.thread:
            if self.thread.is_alive():
                raise ProgressiveError('Scheduler already running')
            else:
                self.thread.start()
        else:
            super(MTScheduler,self).run()

    def stop(self):
        super(MTScheduler,self).stop()

    def _add_module(self, module):
        with self.lock:
            super(MTScheduler,self)._add_module(module)
    def _remove_module(self, module):
        with self.lock:
            super(MTScheduler,self)._remove_module(module)
        
