from progressive.core.common import ProgressiveError

from timeit import default_timer
from toposort import toposort_flatten

import logging
logger = logging.getLogger(__name__)

__all__ = ['Scheduler']

class Scheduler(object):
    def __init__(self):
        self._modules = dict()
        self._running = False
        self._runorder = None
        self._stopped = False
        self._valid = False
        self._start = None

    def timer(self):
        if self._start is None:
            self._start = default_timer()
            return 0
        return default_timer()-self._start

    def collect_dependencies(self):
        dependencies = {}
        for (mid, module) in self._modules.iteritems():
            outs = [m.output_module.id() for m in module.input_slot_values() if m]
            if outs:
                dependencies[mid] = set(outs)
        return dependencies

    def validate(self):
        if not self._valid:
            for module in self._modules.values():
                module.validate()
            self._valid = True

    def is_valid(self):
        return self._valid

    def invalidate(self):
        self._valid = False

    def _before_run(self, run_number):
        pass

    def _after_run(self, run_number):
        pass

    def run(self):
        self._stopped = False
        self._running = True
        self.validate()
        self._runorder = toposort_flatten(self.collect_dependencies())
        logger.info("Scheduler run order: %s", self._runorder)
        done = False
        run_number = 0

        modules = map(self.module, self._runorder)
        for module in modules:
            module.start()
        while not done and not self._stopped:
            run_number += 1
            done = True
            self._before_run(run_number)
            for module in modules:
                if not module.is_ready():
                    if module.is_terminated():
                        logger.info("Module %s terminated", module.id())
                    else:
                        logger.info("Module %s not ready", module.id())
                    continue
                logger.info("Running module %s", module.id())
                module.run(run_number)
                logger.info("Module %s returned", module.id())
                if not module.is_terminated():
                    done = False
                else:
                    logger.info("Module %s terminated", module.id())
            self._after_run(run_number)
        for module in reversed(modules):
            module.end()
        self._running = False

    def stop(self):
        self._stopped = True

    def is_running(self):
        return self._running

    def __len__(self):
        return len(self._modules)

    def exists(self, id):
        return id in self._modules

    def add(self, module):
        if not module.is_created():
            raise ProgressiveError('Cannot add running module %s', module.id())
        self._modules[module.id()] = module

    def module(self, id):
        return self._modules.get(id, None)

    def remove(self, module):
        self.stop()
        module._stop()
        del self._modules[module.id()]

    def modules(self):
        return self._modules


if not hasattr(Scheduler,'default'):
    Scheduler.default = Scheduler()

