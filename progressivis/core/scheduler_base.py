from __future__ import absolute_import, division, print_function

from progressivis.core.utils import ProgressiveError
from progressivis.core.utils import AttributeDict
from progressivis.core.synchronized import synchronized
import functools
from copy import copy
from collections import deque
from timeit import default_timer
from toposort import toposort_flatten
from contextlib import contextmanager
from uuid import uuid4

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np

import logging
import six
logger = logging.getLogger(__name__)

__all__ = ['BaseScheduler']


# Redefined in utils.py?
class FakeLock(object):
    def acquire(self, blocking=1):
        _ = blocking  # keeps pylint happy
        return False

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class BaseScheduler(object):
    default = None
    _last_id = 0

    @classmethod
    def or_default(cls, scheduler):
        return scheduler or cls.default

    def __init__(self, interaction_latency=0.1):
        if interaction_latency <= 0:
            raise ProgressiveError('Invalid interaction_latency, '
                                   'should be strictly positive: %s',
                                   interaction_latency)
        self._lock = self.create_lock()
        # same as clear below
        with self.lock:
            BaseScheduler._last_id += 1
            self._id = BaseScheduler._last_id
        self._modules = dict()
        self._module = AttributeDict(self._modules)
        self._running = False
        self._runorder = None
        self._stopped = False
        self._valid = False
        self._start = None
        self._run_number = 0
        self._run_number_time = {}
        self._tick_proc = None
        self._oneshot_tick_procs = []
        self._idle_proc = None
        self._new_modules_ids = []
        self._slots_updated = False
        self._run_queue = deque()
        self._input_triggered = {}
        self._module_selection = None
        self._selection_target_time = -1
        self.interaction_latency = interaction_latency
        self._reachability = {}
        # Create Sentinel last since it needs the scheduler to be ready
        from .sentinel import Sentinel
        self._sentinel = Sentinel(scheduler=self)

    def create_lock(self):
        return FakeLock()

    def join(self):
        pass

    def clear(self):
        self._modules = dict()
        self._module = AttributeDict(self._modules)
        self._running = False
        self._runorder = None
        self._stopped = False
        self._valid = False
        self._start = None
        self._run_number = 0
        self._run_number_time = {}
        self._tick_proc = None
        self._oneshot_tick_procs = []
        self._idle_proc = None
        self._new_modules_ids = []
        self._slots_updated = False
        self._run_queue = deque()
        self._input_triggered = {}
        self._module_selection = None
        self._selection_target_time = -1
        self._reachability = {}

    @property
    def lock(self):
        return self._lock

    @property
    def id(self):
        return str(self._id)

    def timer(self):
        if self._start is None:
            self._start = default_timer()
            return 0
        return default_timer()-self._start

    @synchronized
    def collect_dependencies(self, only_required=False):
        dependencies = {}
        for (mid, module) in six.iteritems(self._modules):
            if not module.is_valid() or module == self._sentinel:
                continue
            outs = [m.output_module.id for m in module.input_slot_values()
                    if m and (not only_required or
                              module.input_slot_required(m.input_name))]
            dependencies[mid] = set(outs)
        return dependencies

    def compute_reachability(self, dependencies):
        # TODO implement a recursive transitive_closure computation
        # instead of using the brute-force djikstra algorithm
        d = dependencies
        k = list(d.keys())
        n = len(k)
        index = dict(zip(k, range(len(k))))
        row = []
        col = []
        data = []
        for (v1, vs) in six.iteritems(d):
            for v2 in vs:
                col.append(index[v1])
                row.append(index[v2])
                data.append(1)
        mat = csr_matrix((data, (row, col)), shape=(n, n))
        dist = shortest_path(mat, directed=True, return_predecessors=False,
                             unweighted=True)
        self._reachability = {}
        reach_no_vis = set()
        all_vis = set(self.get_visualizations())
        for i1 in range(n):
            v1 = k[i1]
            s = {v1}
            for i2 in range(n):
                v2 = k[i2]
                dst = dist[i1, i2]
                if dst != 0 and dst != np.inf:
                    s.add(v2)
            self._reachability[v1] = s
            if not all_vis.intersection(s):
                logger.debug('No visualization after module %s: %s', v1, s)
                reach_no_vis.update(s)
                if not self.module[v1].is_visualization():
                    reach_no_vis.add(v1)
        logger.debug('Module(s) %s always after visualizations', reach_no_vis)
        # filter out module that reach no vis
        for (k, v) in six.iteritems(self._reachability):
            v.difference_update(reach_no_vis)
        logger.debug('reachability map: %s', self._reachability)

    def get_visualizations(self):
        return [m.id for m in self.modules().values() if m.is_visualization()]

    def get_inputs(self):
        return [m.id for m in self.modules().values() if m.is_input()]

    def reachable_from_inputs(self, inputs):
        reachable = set()
        if len(inputs) == 0:
            return set()
        # collect all modules reachable from the modified inputs
        for i in inputs:
            reachable.update(self._reachability[i])
        all_vis = self.get_visualizations()
        reachable_vis = reachable.intersection(all_vis)
        if reachable_vis:
            # TODO remove modules following visualizations
            return reachable
        return None

    def order_modules(self):
        runorder = None
        try:
            dependencies = self.collect_dependencies()
            runorder = toposort_flatten(dependencies)
            self.compute_reachability(dependencies)
        except ValueError:  # cycle, try to break it then
            # if there's still a cycle, we cannot run the first cycle
            logger.info('Cycle in module dependencies, '
                        'trying to drop optional fields')
            dependencies = self.collect_dependencies(only_required=True)
            runorder = toposort_flatten(dependencies)
            self.compute_reachability(dependencies)
        return runorder

    @staticmethod
    def module_order(x, y):
        if 'order' in x:
            if 'order' in y:
                return x['order']-y['order']
            return 1
        if 'order' in y:
            return -1
        return 0

    def run_queue_length(self):
        return len(self._run_queue)

    def to_json(self, short=True):
        msg = {}
        mods = {}
        with self.lock:
            for (name, module) in six.iteritems(self.modules()):
                mods[name] = module.to_json(short=short)
        mods = mods.values()
        modules = sorted(mods, key=functools.cmp_to_key(self.module_order))
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
        if tick_proc:
            self._tick_proc = tick_proc
        if idle_proc:
            self._idle_proc = idle_proc
        self.run()

    def _step_proc(self, s, run_number):
        # pylint: disable=unused-argument
        self.stop()

    def step(self):
        self.start(tick_proc=self._step_proc)

    def set_tick_proc(self, tick_proc):
        if tick_proc is None or callable(tick_proc):
            self._tick_proc = tick_proc
        else:
            raise ProgressiveError('value should be callable or None',
                                   tick_proc)

    def add_oneshot_tick_proc(self, tick_proc):
        """
        Add a oneshot function that will be run at the next scheduler tick.
        This is especially useful for setting up module connections.
        """
        if not callable(tick_proc):
            logger.warning("tick_proc should be callable")
        else:
            with self.lock:
                self._oneshot_tick_procs += [tick_proc]

    def set_idle_proc(self, idle_proc):
        if idle_proc is None or callable(idle_proc):
            self._idle_proc = idle_proc
        else:
            raise ProgressiveError('value should be callable or None',
                                   idle_proc)

    def idle_proc(self):
        return self._idle_proc

    def slots_updated(self):
        self._slots_updated = True

    def _next_module(self):
        """Yields a possibly infinite sequence of modules. Handles
        order recomputation and starting logic if needed."""
        while (self._run_queue or self._new_modules_ids or
               self._slots_updated) and not self._stopped:
            if self._new_modules_ids or self._slots_updated:
                # Make a shallow copy of the current run order;
                # if we cannot validate the new state, revert to the copy
                prev_run_queue = copy(self._run_queue)
                for mid in self._new_modules_ids:
                    self._modules[mid].starting()
                self._new_modules_ids = []
                self._slots_updated = False
                with self.lock:
                    self._run_queue.clear()
                    self._runorder = self.order_modules()
                    i = 0
                    for mid in self._runorder:
                        m = self._modules[mid]
                        if m is not self._sentinel:
                            self._run_queue.append(m)
                            m.order = i
                            i += 1
                    self._sentinel.order = i
                    self._run_queue.append(self._sentinel)  # always at the end
                if not self.validate():
                    logger.error("Cannot validate progressive workflow,"
                                 " reverting to previous")
                    self._run_queue = prev_run_queue
            yield self._run_queue.popleft()

    def _run_loop(self):
        """Main scheduler loop."""
        # pylint: disable=broad-except
        for module in self._next_module():
            if self._stopped:
                break
            if not self._consider_module(module):
                logger.info("Module %s not part of input management",
                            module.id)
            elif not module.is_ready():
                logger.info("Module %s not ready", module.id)
            else:
                self._run_number += 1
                with self.lock:
                    if self._tick_proc:
                        logger.debug('Calling tick_proc')
                        try:
                            self._tick_proc(self, self._run_number)
                        except Exception as e:
                            logger.warning(e)
                    for proc in self._oneshot_tick_procs:
                        try:
                            proc()
                        except Exception as e:
                            logger.warning(e)
                    self._oneshot_tick_procs = []
                    logger.debug("Running module %s", module.id)
                    module.run(self._run_number)
                    logger.debug("Module %s returned", module.id)
            # Do it for all the modules
            if not module.is_terminated():
                self._run_queue.append(module)
            else:
                logger.info('Module %s is terminated', module.id)
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
        return len(self._modules)-1

    def exists(self, moduleid):
        return moduleid in self._modules

    def generate_id(self, prefix):
        # Try to be nice
        for i in range(1, 10):
            mid = '%s_%d' % (prefix, i)
            if mid not in self._modules:
                return mid
        return '%s_%s' % (prefix, uuid4())

    def run_number_time(self, run_number):
        return self._run_number_time[run_number]

    @synchronized
    def add_module(self, module):
        if not module.is_created():
            raise ProgressiveError('Cannot add running module %s', module.id)
        if module.id is None:
            # pylint: disable=protected-access
            module._id = self.generate_id(module.pretty_typename())
        self._add_module(module)

    def _add_module(self, module):
        self._new_modules_ids += [module.id]
        self._modules[module.id] = module

    @property
    def module(self):
        return self._module

    @synchronized
    def remove_module(self, module):
        if isinstance(module, six.string_types):
            module = self.module[module]
        module.terminate()
#            self.stop()
#            module._stop(self._run_number)
        self._remove_module(module)

    def _remove_module(self, module):
        del self._modules[module.id]

    def modules(self):
        return self._modules

    def __getitem__(self, mid):
        return self._modules.get(mid, None)

    def run_number(self):
        return self._run_number

    @contextmanager
    def stdout_parent(self):
        yield

    @synchronized
    def for_input(self, module):
        self._input_triggered[module.id] = 0  # don't know the run number yet
        # limit modules to react
        sel = self._reachability[module.id]
        if sel:
            if self._module_selection is None:
                self._module_selection = sel
                self._selection_target_time = self.timer() + self.interaction_latency
            else:
                self._module_selection.update(sel)
            logger.debug('Input selection for module: %s',
                         self._module_selection)
        return self.run_number()+1

    def has_input(self):
        if self._module_selection is not None:
            if len(self._module_selection) == 0:
                logger.debug('Finishing input management')
                self._module_selection = None
                self._selection_target_time = -1
            else:
                return True
        return False

    def _consider_module(self, module):
        if not self.has_input():
            return True
        if module is self._sentinel:
            return True
        if module.id in self._module_selection:
            self._module_selection.remove(module.id)
            # TODO reset quantum
            logger.debug('Module %s ready for scheduling', module.id)
            return True
        logger.debug('Module %s NOT ready for scheduling', module.id)
        return False

    def time_left(self):
        if self._selection_target_time <= 0:
            logger.error('time_left called with no target time')
            return 0
        return max(0, self._selection_target_time - self.timer())

    def fix_quantum(self, module, quantum):
        if self.has_input() and module.id in self._module_selection:
            quantum = self.time_left() / len(self._module_selection)
            print("Changed quantum to ", quantum)
        if quantum == 0:
            quantum = 0.1
            logger.error('Quantum is 0 in %s, setting it to'
                         ' a reasonable value', module.id)
        return quantum
