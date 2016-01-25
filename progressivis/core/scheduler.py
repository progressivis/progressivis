from progressivis.core.common import ProgressiveError
from progressivis.core.utils import AttributeDict

from collections import deque
from time import sleep
from timeit import default_timer
from toposort import toposort_flatten
from contextlib import contextmanager
from uuid import uuid4

import logging
logger = logging.getLogger(__name__)

__all__ = ['Scheduler']

class FakeLock(object):
    def acquire(self, blocking=1):
        return False
    def release(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return

class Scheduler(object):
    default = None
    
    def __init__(self):
        self._lock = self.create_lock()
        self._modules = dict()
        self._module = AttributeDict(self._modules)
        self._running = False
        self._runorder = None
        self._stopped = False
        self._valid = False
        self._start = None
        self._run_number = 0
        self._run_number_time =  {}
        self._tick_proc = None
        self._idle_proc = None
        self._new_modules_ids = []
        self._slots_updated = False
        self._run_queue = deque()

    def create_lock(self):
        #import traceback
        #print 'creating lock'
        #traceback.print_stack()
        return FakeLock()

    def clear(self):
        self._modules = dict()
        self._module = AttributeDict(self._modules)
        self._running = False
        self._runorder = None
        self._stopped = False
        self._valid = False
        self._start = None
        self._run_number = 0
        self._run_number_time =  {}
        self._tick_proc = None
        self._idle_proc = None
        self._new_modules_ids = []
        self._slots_updated = False
        self._run_queue = deque()

    @property
    def lock(self):
        return self._lock

    def timer(self):
        if self._start is None:
            self._start = default_timer()
            return 0
        return default_timer()-self._start

    def collect_dependencies(self, only_required=False):
        dependencies = {}
        with self.lock:
            for (mid, module) in self._modules.iteritems():
                if not module.is_valid(): # ignore invalid modules
                    continue
                outs = [m.output_module.id for m in module.input_slot_values() \
                    if m and (not only_required or module.input_slot_required(m.input_name)) ]
                dependencies[mid] = set(outs)
        return dependencies

    def order_modules(self):
        runorder = None
        try:
            runorder = toposort_flatten(self.collect_dependencies())
        except ValueError: # cycle, try to break it then
            # if there's still a cycle, we cannot run the first cycle
            logger.info('Cycle in module dependencies, trying to drop optional fields')
            runorder = toposort_flatten(self.collect_dependencies(), only_required=True)
        return runorder

    @staticmethod
    def module_order(x,y):
        if 'order' in x:
            if 'order' in y:
                return x['order']-y['order']
            return 1
        if 'order' in y:
            return -1
        return 0

    def to_json(self, short=True):
        msg = {}
        mods = {}
        for (name,module) in self.modules().iteritems():
            mods[name] = module.to_json(short=short)
                           
        if self._runorder:
            i = 0
            for m in self._runorder:
                if m in mods:
                    mods[m]['order'] = i
                else:
                    logger.error("module '%s' not in module list", m)
                    mods[m] = {'module': None, 'order': i }
                i += 1
        mods = mods.values()
        modules = sorted(mods, self.module_order)
        msg['modules'] = modules
        msg['is_valid'] = self.is_valid()
        msg['is_running'] = self.is_running()
        msg['is_terminated'] = self.is_terminated()
        msg['run_number'] = self.run_number()
        msg['status'] = 'success'
        return msg

    def validate(self):
        if not self._valid:
            valid = True
            for module in self._modules.values():
                if not module.validate():
                    logger.error('Cannot validate module %s', module.id)
                    valid = False
            self._valid = valid
        return self._valid

    def is_valid(self):
        return self._valid

    def invalidate(self):
        self._valid = False

    def _before_run(self):
        pass

    def _after_run(self):
        pass

    def start(self, tick_proc=None, idle_proc=None):
        self._tick_proc = tick_proc
        self._idle_proc = idle_proc
        self.run()

    def _step_proc(self, s, run_number):
        self.stop()

    def step(self):
        self.start(tick_proc=self._step_proc)

    def set_tick_proc(self, tick_proc):
        if tick_proc is None or callable(tick_proc):
            self._tick_proc = tick_proc
        else:
            raise ProgressiveError('value should be callable or None', tick_proc)
    def set_idle_proc(self, idle_proc):
        if idle_proc is None or callable(idle_proc):
            self.idle_proc = idle_proc
        else:
            raise ProgressiveError('value should be callable or None', idle_proc)

    def slots_updated(self):
        self._slots_updated = True

    def next_module(self):
        """Yields a possibly infinite sequence of modules. Handles 
        order recomputation and starting logic if needed."""
        while (self._run_queue or self._new_modules_ids or self._slots_updated) \
               and not self._stopped:
            if self._new_modules_ids or self._slots_updated:
                for id in self._new_modules_ids:
                    self._modules[id].starting()
                self._new_modules_ids = []
                self._slots_updated = False
                with self.lock:
                    self._run_queue.clear()
                    self._runorder = self.order_modules() 
                    for id in self._runorder:
                        self._run_queue.append(self._modules[id])
                if not self.validate():
                    raise ProgressiveError('Cannot validate progressive workflow')
            yield self._run_queue.popleft()
            

    def _run_loop(self):
        """Main scheduler loop."""
        for module in self.next_module():
             if self._stopped:
                 break
             if not module.is_ready():
                 logger.info("Module %s not ready", module.id)
                 continue
             # Note: is calling tick_prock for each module the right thing?
             # Should we have a sentry in the run queue to invoke ticks?
             if self._tick_proc:
                 self._tick_proc(self, self._run_number)
             with self.lock:
                 self._run_number += 1
             logger.info("Running module %s", module.id)
             module.run(self._run_number)
             logger.info("Module %s returned", module.id)
             if not module.is_terminated():
                 self._run_queue.append(module)
             # TODO: idle sleep, cleanup_run

    def run(self):
        self._stopped = False
        self._running = True
        self._before_run()

        self._run_loop()

        modules = [self.module[m] for m in self._runorder]
        for module in reversed(modules):
            module.ending()
        self._running = False
        self._stopped = True
        self.done()

    def stop(self):
        self._stopped = True

    def is_running(self):
        return self._running

    def is_terminated(self):
        for m in self.modules().values():
            if not m.is_terminated():
                return False
        return True

    def done(self):
        pass

    def __len__(self):
        return len(self._modules)

    def exists(self, id):
        return id in self._modules

    def generate_id(self, prefix):
        # Try to be nice
        for i in range(1,10):
            mid = '%s_%d' % (prefix, i)
            if mid not in self._modules:
                return mid
        return '%s_%s' % (prefix, uuid4())

    def run_number_time(self, run_number):
        return self._run_number_time[run_number]

    def add_module(self, module):
        with self.lock:
            if not module.is_created():
                raise ProgressiveError('Cannot add running module %s', module.id)
            if module.id is None:
                module._id = self.generate_id(module.pretty_typename())
            self._add_module(module)

    def _add_module(self, module):
         self._new_modules_ids += [module.id]
         self._modules[module.id] = module

    @property
    def module(self):
        return self._module

    def remove_module(self, module):
        with self.lock:
            if isinstance(module,str) or isinstance(module,unicode):
                module = self.module[module]
            self.stop()
            module._stop(self._run_number)
            self._remove_module(module)

    def _remove_module(self, module):
        del self._modules[module.id]

    def modules(self):
        return self._modules

    def run_number(self):
        return self._run_number

    @contextmanager
    def stdout_parent(self):
        yield


if Scheduler.default is None:
    Scheduler.default = Scheduler()

