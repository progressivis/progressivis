from progressive.core.common import ProgressiveError

from timeit import default_timer
from toposort import toposort_flatten

import logging
logger = logging.getLogger(__name__)

__all__ = ['Scheduler']

class Scheduler(object):
    default = None
    
    def __init__(self):
        self._modules = dict()
        self._running = False
        self._runorder = None
        self._stopped = False
        self._valid = False
        self._start = None
        self._run_number = 0

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
            outs = [m.output_module.id() for m in module.input_slot_values() \
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

    def start(self):
        self.run()

    def run(self):
        self._stopped = False
        self._running = True
        self.validate()
        done = False

        self._runorder = self.order_modules()
        logger.info("Scheduler run order: %s", self._runorder)
        modules = map(self.module, self._runorder)
        for module in modules:
            module.starting()
        while not done and not self._stopped:
            self._run_number += 1
            done = True
            self._before_run()
            for module in modules:
                if not module.is_ready():
                    if module.is_terminated():
                        logger.info("Module %s terminated", module.id())
                    else:
                        logger.info("Module %s not ready", module.id())
                    continue
                logger.info("Running module %s", module.id())
                module.run(self._run_number)
                logger.info("Module %s returned", module.id())
                if not module.is_terminated():
                    done = False
                else:
                    logger.info("Module %s terminated", module.id())
            self._after_run()
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

    def add(self, module):
        if not module.is_created():
            raise ProgressiveError('Cannot add running module %s', module.id())
        self._add_module(module)

    def _add_module(self, module):
        self._modules[module.id()] = module

    def module(self, id):
        return self._modules.get(id, None)

    def remove(self, module):
        self.stop()
        module._stop(self._run_number)
        self._remove_module(module)

    def _remove_module(self, module):
        del self._modules[module.id()]

    def modules(self):
        return self._modules


if Scheduler.default is None:
    Scheduler.default = Scheduler()

