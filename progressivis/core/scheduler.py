from progressivis.core.common import ProgressiveError
from progressivis.core.utils import AttributeDict

from timeit import default_timer
from toposort import toposort_flatten
from contextlib import contextmanager
from uuid import uuid4

import logging
logger = logging.getLogger(__name__)

__all__ = ['Scheduler']

class Scheduler(object):
    default = None
    
    def __init__(self):
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

    def timer(self):
        if self._start is None:
            self._start = default_timer()
            return 0
        return default_timer()-self._start

    def collect_dependencies(self, only_required=False):
        dependencies = {}
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

    def to_json(self):
        msg = {}
        mods = {}
        for (name,module) in self.modules().iteritems():
            mods[name] = module.to_json()
        modules = []
        if self._runorder:
            i = 0
            for m in self._runorder:
                if m in mods:
                    mods[m]['order'] = i
                else:
                    logger.error("module '%s' not in module list", m)
                    mods[m] = {'module': None, 'order': i }
                    modules.append(mods[m])
                    i += 1
        else:
            modules = mods.values()
        msg['modules'] = modules
        msg['is_valid'] = self.is_valid()
        msg['is_running'] = self.is_running()
        msg['run_number'] = self.run_number()
        msg['status'] = 'success'
        return msg

    def validate(self):
        if not self._valid:
            valid = True
            for module in self._modules.values():
                if not module.validate():
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

    def start(self, tick_proc=None):
        self._tick_proc = tick_proc
        self.run()

    def run(self):
        self._stopped = False
        self._running = True
        if not self.validate():
            raise ProgressiveError('Cannot validate progressive workflow')

        self._runorder = self.order_modules()
        logger.info("Scheduler run order: %s", self._runorder)
        modules = [self.module[m] for m in self._runorder]

        self._run_number_time[self._run_number] = self.timer()
        for module in modules:
            module.starting()
        while len(modules)!=0 and not self._stopped:
            self._run_number += 1
            if self._tick_proc:
                self._tick_proc(self, self._run_number)
            self._before_run()
            for module in modules:
                if not module.is_ready():
                    logger.info("Module %s not ready", module.id)
                    continue
                logger.info("Running module %s", module.id)
                module.run(self._run_number)
                logger.info("Module %s returned", module.id)
            for module in modules:
                module.cleanup_run(self._run_number)
            # remove terminated modules from the run queue
            modules = [m for m in modules if not m.is_terminated()]
            self._after_run()
            self._run_number_time[self._run_number] = self.timer()

        # get back all the modules
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
        if not module.is_created():
            raise ProgressiveError('Cannot add running module %s', module.id)
        if module.id is None:
            module._id = self.generate_id(module.pretty_typename())
        self._add_module(module)

    def _add_module(self, module):
        self._modules[module.id] = module

    @property
    def module(self):
        return self._module

    def remove_module(self, module):
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

