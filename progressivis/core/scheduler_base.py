"""
Base Scheduler class, runs progressive modules.
"""
from __future__ import absolute_import, division, print_function
import time
import logging
import functools
#from copy import copy
#from collections import deque
from collections import Iterable
from timeit import default_timer
import six

# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import shortest_path
# import numpy as np


from .utils import ProgressiveError, FakeLock, Condition
from .dataflow import Dataflow
from .synchronized import synchronized
#from .toposort import toposort


logger = logging.getLogger(__name__)

__all__ = ['BaseScheduler']

KEEP_RUNNING = 5

class _InteractionOpts(object):
    def __init__(self, starving_mods=None, max_time=None, max_iter=None):
        # TODO: checks ...
        self.starving_mods = starving_mods
        self.max_time = max_time
        self.max_iter = max_iter


class BaseScheduler(object):
    "Base Scheduler class, runs progressive modules"
    # pylint: disable=too-many-public-methods,too-many-instance-attributes
    default = None
    _last_id = 0
    @classmethod
    def or_default(cls, scheduler):
        "Return the specified scheduler of, in None, the default one."
        return scheduler or cls.default

    def __init__(self, interaction_latency=1):
        if interaction_latency <= 0:
            raise ProgressiveError('Invalid interaction_latency, '
                                   'should be strictly positive: %s'% interaction_latency)
        self._lock = self.create_lock()
        # same as clear below
        with self.lock:
            BaseScheduler._last_id += 1
            self._name = BaseScheduler._last_id
        self._modules = {}
        self._dependencies = None
        self._running = False
        self._stopped = True
        self._runorder = None
        self._new_modules = None
        self._new_dependencies = None
        self._new_runorder = None
        self._start = None
        self._run_number = 0
        self._tick_procs = []
        self._tick_once_procs = []
        self._idle_procs = []
        self.version = 0
        self._run_list = []
        self._run_index = 0
        self._module_selection = None
        self._selection_target_time = -1
        self.interaction_latency = interaction_latency
        self._reachability = {}
        self._start_inter = 0
        self._inter_cycles_cnt = 0
        self._interaction_opts = None
        self._hibernate_cond = Condition()
        self._keep_running = KEEP_RUNNING

    def set_interaction_opts(self, starving_mods=None, max_time=None, max_iter=None):
        if starving_mods:
            if not isinstance(starving_mods, Iterable):
                raise ValueError("starving_mods must be iterable")
            from .module import Module
            for elt in starving_mods:
                if not isinstance(elt, Module):
                    raise ValueError("starving_mods  requires a list of Modules")
        if max_time:
            if not isinstance(max_time, (int, float)):
                raise ValueError("max_time must be a float or an int")
            if max_time <= 0:
                raise ValueError("max_time must be positive")
        if max_iter:
            if not isinstance(max_iter, int):
                raise ValueError("max_iter must be an int")
            if max_iter <= 0:
                raise ValueError("max_iter must be positive")
        self._interaction_opts = _InteractionOpts(starving_mods, max_time, max_iter)

    def _proc_interaction_opts(self):
        if not self.has_input():
            return
        if self._interaction_opts is None:
            self._module_selection = None
            self._inter_cycles_cnt = 0
            return
        if self._interaction_opts.starving_mods:
            if not sum([mod.steps_acc for mod in self._interaction_opts.starving_mods]):
                print("Exiting shortcut mode because data "
                      "inputs on witnesses are dried",
                      self._interaction_opts.starving_mods)
                self._module_selection = None
                self._inter_cycles_cnt = 0
                return
        if self._interaction_opts.max_time:
            duration = default_timer()-self._start_inter
            if  duration >= self._interaction_opts.max_time:
                print("Exiting shortcut mode on time out, duration: ", duration)
                self._module_selection = None
                self._inter_cycles_cnt = 0
                return
        if self._interaction_opts.max_iter:
            if self._inter_cycles_cnt >= self._interaction_opts.max_iter:
                self._module_selection = None
                self._inter_cycles_cnt = 0
                print("Exiting shortcut mode after ", self._interaction_opts.max_iter, " cycles")
            else:
                self._inter_cycles_cnt += 1

    def create_lock(self):
        "Create a lock, fake in this class, real in the derived Scheduler"
        # pylint: disable=no-self-use
        return FakeLock()

    def join(self):
        "Wait for this execution thread to finish."

    @property
    def lock(self):
        "Return the scheduler lock."
        return self._lock

    def dataflow(self):
        """
        Returns a newly created Dataflow, to use in a context manager.

        """
        return Dataflow(self)

    @property
    def name(self):
        "Return the scheduler id"
        return str(self._name)

    def timer(self):
        "Return the scheduler timer."
        if self._start is None:
            self._start = default_timer()
            return 0
        return default_timer()-self._start

    def run_queue_length(self):
        "Return the length of the run queue"
        return len(self._run_list)

    def to_json(self, short=True):
        "Return a dictionary describing the scheduler"
        msg = {}
        mods = {}
        with self.lock:
            for (name, module) in six.iteritems(self.modules()):
                mods[name] = module.to_json(short=short)
        mods = mods.values()
        modules = sorted(mods, key=functools.cmp_to_key(self._module_order))
        msg['modules'] = modules
        msg['is_running'] = self.is_running()
        msg['is_terminated'] = self.is_terminated()
        msg['run_number'] = self.run_number()
        msg['status'] = 'success'
        return msg


    def _before_run(self):
        pass

    def _after_run(self):
        pass

    def start(self, tick_proc=None, idle_proc=None):
        "Start the scheduler."
        if tick_proc:
            self._tick_procs = []
            self.on_tick(tick_proc)
        if idle_proc:
            self._idle_procs = []
            self.on_idle(idle_proc)
        self.run()

    def _step_proc(self, s, run_number):
        # pylint: disable=unused-argument
        self.stop()

    def step(self):
        "Start the scheduler for on step."
        self.start(tick_proc=self._step_proc)

    def on_tick(self, tick_proc):
        "Set a procedure to call at each tick."
        assert callable(tick_proc)
        self._tick_procs.append(tick_proc)

    def remove_tick(self, tick_proc):
        "Remove a tick callback"
        self._tick_procs.remove(tick_proc)

    def on_tick_once(self, tick_proc):
        """
        Add a oneshot function that will be run at the next scheduler tick.
        This is especially useful for setting up module connections.
        """
        assert callable(tick_proc)
        self._tick_once_procs.append(tick_proc)

    def remove_tick_once(self, tick_proc):
        "Remove a tick once callback"
        self._tick_once_procs.remove(tick_proc)

    def on_idle(self, idle_proc):
        "Set a procedure that will be called when there is nothing else to do."
        assert callable(idle_proc)
        self._idle_procs.append(idle_proc)

    def remove_idle(self, idle_proc):
        "Remove an idle callback."
        assert callable(idle_proc)
        self._idle_procs.remove(idle_proc)

    def run(self):
        "Run the modules, called by start()."
        self._stopped = False
        self._running = True
        self._start = default_timer()
        self._before_run()

        self._run_loop()

        modules = [self._modules[m] for m in self._runorder]
        for module in reversed(modules):
            module.ending()
        self._running = False
        self._stopped = True
        self.done()

    def _run_loop(self):
        """Main scheduler loop."""
        # pylint: disable=broad-except
        for module in self._next_module():
            if self.no_more_data() and self.all_blocked() and self.is_waiting_for_input():
                if not self._keep_running:
                    with self._hibernate_cond:
                        self._hibernate_cond.wait()
            if self._keep_running:
                self._keep_running -= 1
            if not (self._consider_module(module) and (module.is_ready() or self.has_input())):
                continue
            self._run_number += 1
            with self.lock:
                self._run_tick_procs()
                module.run(self._run_number)

    def _next_module(self):
        """Yields a possibly infinite sequence of modules.
        Handles order recomputation and starting logic if needed.
        """
        self._run_index = 0
        first_run = self._run_number
        input_mode = self.has_input()
        self._start_inter = 0
        self._inter_cycles_cnt = 0
        while not self._stopped:
            # Apply changes in the dataflow
            if self._new_modules:
                self._update_modules()
                self._run_index = 0
                first_run = self._run_number
            # If run_list empty, we're done
            if not self._run_list:
                break
            # Check for interactive input mode
            if input_mode != self.has_input():
                if input_mode: # end input mode
                    print('Ending interactive mode after', default_timer()-self._start_inter)
                    self._start_inter = 0
                    self._inter_cycles_cnt = 0
                    input_mode = False
                else:
                    self._start_inter = default_timer()
                    print('Starting interactive mode at', self._start_inter)
                    input_mode = True
                # Restart from beginning
                self._run_index = 0
                first_run = self._run_number
            module = self._run_list[self._run_index]
            self._run_index += 1 # allow it to be reset
            yield module
            if self._run_index >= len(self._run_list): # end of modules
                self._end_of_modules(first_run)
                first_run = self._run_number

    def all_blocked(self):
        "Return True if all the modules are blocked, False otherwise"
        from .module import Module
        for module in self._run_list:
            if module.state != Module.state_blocked:
                return False
        return True

    def is_waiting_for_input(self):
        "Return True if there is at least one input module"
        for module in self._run_list:
            if module.is_input():
                return True
        return False

    def no_more_data(self):
        "Return True if at least one module has data input."
        for module in self._run_list:
            if module.is_data_input():
                return False
        return True

    def _commit(self, dataflow):
        assert dataflow.version == self.version
        self.version += 1
        self._new_modules = dataflow.modules()
        self._new_dependencies = dataflow.inputs
        self._new_runorder = dataflow.order_modules()
        # The slots in the module,_modules, and _runorder will be updated
        # in _update_modules when the scheduler decides it is time to do so.

    def _update_modules(self):
        if not self._new_modules:
            return
        #import pdb; pdb.set_trace()
        prev_keys = set(self._modules.keys())
        modules = {module.name: module for module in self._new_modules}
        keys = set(modules.keys())
        added = keys - prev_keys
        deleted = prev_keys - keys
        if deleted:
            logger.info("Scheduler deleted modules %s", deleted)
            for mid in deleted:
                self._modules[mid].ending()
        self._modules = modules
        if added:
            logger.info("Scheduler adding modules %s", added)
            for mid in added:
                modules[mid].starting()
        if not(deleted or added):
            logger.info("Scheduler updated with no new module(s)")
        self._dependencies = self._new_dependencies
        self._new_dependencies = None
        for mid, slots in six.iteritems(self._dependencies):
            modules[mid].reconnect(slots)
        self._new_modules = None
        with self.lock:
            self._run_list = []
            self._runorder = self._new_runorder
            self._new_runorder = None
            for i, mid in enumerate(self._runorder):
                module = self._modules[mid]
                self._run_list.append(module)
                module.order = i

    def _end_of_modules(self, first_run):
        # Reset interaction mode
        #import pdb;pdb.set_trace()
        self._proc_interaction_opts()
        self._selection_target_time = -1
        new_list = [m for m in self._run_list if not m.is_terminated()]
        self._run_list = new_list
        if first_run == self._run_number: # no module ready
            has_run = False
            for proc in self._idle_procs:
                #pylint: disable=broad-except
                try:
                    logger.debug('Running idle proc')
                    proc(self, self._run_number)
                    has_run = True
                except Exception as exc:
                    logger.error(exc)
            if not has_run:
                logger.info('sleeping %f', 0.2)
                time.sleep(0.2)
        self._run_index = 0

    def _run_tick_procs(self):
        #pylint: disable=broad-except
        for proc in self._tick_procs:
            logger.debug('Calling tick_proc')
            try:
                proc(self, self._run_number)
            except Exception as exc:
                logger.warning(exc)
        for proc in self._tick_once_procs:
            try:
                proc()
            except Exception as exc:
                logger.warning(exc)
            self._tick_once_procs = []

    def stop(self):
        "Stop the execution."
        with self._hibernate_cond:
            self._keep_running = KEEP_RUNNING
            self._hibernate_cond.notify()
        self._stopped = True

    def is_running(self):
        "Return True if the scheduler is currently running."
        return self._running

    def is_terminated(self):
        "Return True if the scheduler is terminated."
        for module in self.modules().values():
            if not module.is_terminated():
                return False
        return True

    def done(self):
        "Called when the execution is done. Can be overridden in subclasses."
        pass

    def __len__(self):
        return len(self._modules)

    def exists(self, moduleid):
        "Return True if the moduleid exists in this scheduler."
        return moduleid in self._modules

    def modules(self):
        "Return the dictionary of modules."
        return self._modules

    def __getitem__(self, mid):
        return self._modules.get(mid, None)

    def run_number(self):
        "Return the last run number."
        return self._run_number

    @synchronized
    def for_input(self, module):
        """
        Notify this scheduler that the module has received input
        that should be served fast.
        """
        with self._hibernate_cond:
            self._keep_running = KEEP_RUNNING
            self._hibernate_cond.notify()
        sel = self._reachability[module.name]
        if sel:
            if not self._module_selection:
                logger.info('Starting input management')
                self._module_selection = set(sel)
                self._selection_target_time = (self.timer() +
                                               self.interaction_latency)
            else:
                self._module_selection.update(sel)
            logger.debug('Input selection for module: %s',
                         self._module_selection)
            print('Input selection for module: %s'%self._module_selection)
        return self.run_number()+1

    def has_input(self):
        "Return True of the scheduler is in input mode"
        if self._module_selection is None:
            return False
        if not self._module_selection: # empty, cleanup
            logger.info('Finishing input management')
            self._module_selection = None
            self._selection_target_time = -1
            return False
        return True

    def _consider_module(self, module):
        if not self.has_input():
            return True
        if module.name in self._module_selection:
            #self._module_selection.remove(module.name)
            logger.debug('Module %s ready for scheduling', module.name)
            return True
        logger.debug('Module %s NOT ready for scheduling', module.name)
        return False

    def time_left(self):
        "Return the time left to run for this slot."
        if self._selection_target_time <= 0 and not self.has_input():
            logger.error('time_left called with no target time')
            return 0
        return max(0, self._selection_target_time - self.timer())

    def fix_quantum(self, module, quantum):
        "Fix the quantum of the specified module"
        if self.has_input() and module.name in self._module_selection:
            quantum = self.time_left() / len(self._module_selection)
        if quantum == 0:
            quantum = 0.1
            logger.info('Quantum is 0 in %s, setting it to'
                        ' a reasonable value', module.name)
        return quantum

    def close_all(self):
        "Close all the resources associated with this scheduler."
        for mod in self.modules().values():
            # pylint: disable=protected-access
            if (hasattr(mod, '_table') and
                    mod._table is not None and
                    mod._table.storagegroup is not None):
                mod._table.storagegroup.close_all()
            if (hasattr(mod, '_params') and
                    mod._params is not None and
                    mod._params.storagegroup is not None):
                mod._params.storagegroup.close_all()
            if (hasattr(mod, 'storagegroup') and
                    mod.storagegroup is not None):
                mod.storagegroup.close_all()

    # def order_modules(self):
    #     """Compute a topological order for the modules.
    #     Should do something smarted with exceptions.
    #     """
    #     runorder = None
    #     try:
    #         dependencies = self._collect_dependencies()
    #         runorder = toposort(dependencies)
    #         #print('normal order of', dependencies, 'is', runorder, file=sys.stderr)
    #         self._compute_reachability(dependencies)
    #     except ValueError:  # cycle, try to break it then
    #         # if there's still a cycle, we cannot run the first cycle
    #         # TODO fix this
    #         logger.info('Cycle in module dependencies, '
    #                     'trying to drop optional fields')
    #         dependencies = self._collect_dependencies(only_required=True)
    #         runorder = toposort(dependencies)
    #         #print('Filtered order of', dependencies, 'is', runorder, file=sys.stderr)
    #         self._compute_reachability(dependencies)
    #     return runorder

    # @synchronized
    # def _collect_dependencies(self, only_required=False):
    #     dependencies = {}
    #     for (mid, module) in six.iteritems(self._modules):
    #         if not module.is_valid():
    #             continue
    #         outs = [m.output_module.name for m in module.input_slot_values()
    #                 if m and (not only_required or
    #                           module.input_slot_required(m.input_name))]
    #         dependencies[mid] = set(outs)
    #     return dependencies

    # def _compute_reachability(self, dependencies):
    #     # pylint: disable=too-many-locals
    #     k = list(dependencies.keys())
    #     size = len(k)
    #     index = dict(zip(k, range(size)))
    #     row = []
    #     col = []
    #     data = []
    #     for (vertex1, vertices) in six.iteritems(dependencies):
    #         for vertex2 in vertices:
    #             col.append(index[vertex1])
    #             row.append(index[vertex2])
    #             data.append(1)
    #     mat = csr_matrix((data, (row, col)), shape=(size, size))
    #     dist = shortest_path(mat,
    #                          directed=True,
    #                          return_predecessors=False,
    #                          unweighted=True)
    #     self._reachability = {}
    #     reach_no_vis = set()
    #     all_vis = set(self.get_visualizations())
    #     for index1 in range(size):
    #         vertex1 = k[index1]
    #         s = {vertex1}
    #         for index2 in range(size):
    #             vertex2 = k[index2]
    #             dst = dist[index1, index2]
    #             if dst not in (0, np.inf):
    #                 s.add(vertex2)
    #         self._reachability[vertex1] = s
    #         if not all_vis.intersection(s):
    #             logger.info('No visualization after module %s: %s', vertex1, s)
    #             reach_no_vis.update(s)
    #             if not self.module[vertex1].is_visualization():
    #                 reach_no_vis.add(vertex1)
    #     logger.info('Module(s) %s always after visualizations', reach_no_vis)
    #     # filter out module that reach no vis
    #     for (k, v) in six.iteritems(self._reachability):
    #         v.difference_update(reach_no_vis)
    #     logger.info('reachability map: %s', self._reachability)

    @staticmethod
    def _module_order(x, y):
        if 'order' in x:
            if 'order' in y:
                return x['order']-y['order']
            return 1
        if 'order' in y:
            return -1
        return 0

