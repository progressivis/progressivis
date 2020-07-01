"""
Base Scheduler class, runs progressive modules.
"""
import logging
import functools
from collections.abc import Iterable
from timeit import default_timer
import time
from .dataflow import Dataflow
import progressivis.core.aio as aio

from progressivis.utils.errors import ProgressiveError

logger = logging.getLogger(__name__)

__all__ = ['Scheduler']

KEEP_RUNNING = 5
SHORTCUT_TIME = 1.5

class Scheduler(object):
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
                                   'should be strictly positive: %s'
                                   % interaction_latency)

        # same as clear below
        Scheduler._last_id += 1
        self._name = Scheduler._last_id
        self._modules = {}
        self._dependencies = None
        self._running = False
        self._stopped = True
        self._runorder = None
        self._new_modules = None
        self._new_dependencies = None
        self._new_runorder = None
        self._new_reachability = None
        self._start = None
        self._step_once = False
        self._exit = False
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
        self._hibernate_cond = None
        self._keep_running = KEEP_RUNNING
        self.dataflow = Dataflow(self)
        self.module_iterator = None
        self._enter_cnt = 1
        self._lock = None
        self._task = None
        #self.runners = set()
        self.shortcut_evt = None
        self.coros = []

    async def shortcut_manager(self):
        if self.shortcut_evt is None:
            self.shortcut_evt = aio.Event()
        while True:
            await self.shortcut_evt.wait()
            await aio.sleep(SHORTCUT_TIME)
            self._module_selection = None
            self.shortcut_evt.clear()
            
    def new_run_number(self):
        self._run_number += 1
        return self._run_number

    def __enter__(self):
        if self.dataflow is None:
            self.dataflow = Dataflow(self)
            self._enter_cnt = 1
        else:
            self._enter_cnt += 1
        return self.dataflow

    def __exit__(self, exc_type, exc_value, traceback):
        self._enter_cnt -= 1
        if exc_type is None:
            if self._enter_cnt == 0:
                self._commit(self.dataflow)
                self.dataflow = None
        else:
            logger.info('Aborting Dataflow with exception %s', exc_type)
            if self._enter_cnt == 0:
                self.dataflow.aborted()
                self.dataflow = None

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
        for (name, module) in self.modules().items():
            mods[name] = module.to_json(short=short)
        modules = sorted(mods.values(),
                         key=functools.cmp_to_key(self._module_order))
        msg['modules'] = modules
        msg['is_running'] = self.is_running()
        msg['is_terminated'] = self.is_terminated()
        msg['run_number'] = self.run_number()
        msg['status'] = 'success'
        return msg

    def _repr_html_(self):
        html_head = "<div type='schedule'>"
        html_head = """
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>"""
        html_end = "</div>"
        html_head += """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Id</th><th>Class</th><th>State</th><th>Last Update</th><th>Order</th>
    </tr>
  </thead>
  <tbody>"""
        columns = ['id', 'classname', 'state', 'last_update', 'order']
        for mod in self._run_list:
            values = mod.to_json(short=True)
            html_head += "<tr>"
            html_head += "".join(["<td>%s</td>" %
                                  (values[column]) for column in columns])
        html_end = "</tbody></table>"
        return html_head + html_end

    @staticmethod
    def set_default():
        "Set the default scheduler."
        if not isinstance(Scheduler.default, Scheduler):
            Scheduler.default = Scheduler()

    def _before_run(self):
        logger.debug("Before run %d", self._run_number)

    def _after_run(self):
        pass

    async def start(self, tick_proc=None, idle_proc=None, coros=()):

        if self._lock is None:
            self._lock = aio.Lock()
        async with self._lock:
            if self._task is not None:
                raise ProgressiveError('Trying to start scheduler task'
                                       ' inside scheduler task')
            
            self._task = True            
        self.coros=list(coros)
        if tick_proc:
            assert callable(tick_proc)
            self._tick_procs = [tick_proc]
        else:
            self._tick_procs = []
        if idle_proc:
            assert callable(idle_proc)
            self._idle_procs = [idle_proc]
        else:
            self._idle_procs = []
        await self.run()

    def task_start(self, *args, **kwargs):
        return aio.create_task(self.start(*args, **kwargs))
    
    def _step_proc(self, s, run_number):
        # pylint: disable=unused-argument
        self.stop()

    async def step(self):
        "Start the scheduler for on step."
        await self.start(tick_proc=self._step_proc)

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

    def idle_proc(self):
        pass

    async def run(self):
        "Run the modules, called by start()."
        global KEEP_RUNNING
        # from .sentinel import Sentinel
        # sl = Sentinel(scheduler=self)
        if self.dataflow:
            assert self._enter_cnt == 1
            self._commit(self.dataflow)
            self.dataflow = None
            self._enter_cnt = 0
        self._stopped = False
        self._running = True
        self._start = default_timer()
        self._before_run()
        # if self._new_modules:
        #    self._update_modules()
        runners = [self._run_loop(), self.shortcut_manager()]
        runners.extend([aio.create_task(coro)
                        for coro in self.coros])
        # runners.append(aio.create_task(self.unlocker(), "unlocker"))
        # TODO: find the "right" initialisation value ...
        KEEP_RUNNING =  min(50, len(self._run_list) * 3)
        self._keep_running = KEEP_RUNNING
        await aio.gather(*runners)
        modules = [self._modules[m] for m in self._runorder]
        for module in reversed(modules):
            module.ending()
        self._running = False
        self._stopped = True
        self._after_run()
        self.done()

    async def _run_loop(self):
        """Main scheduler loop."""
        # pylint: disable=broad-except
        if self._hibernate_cond is None:
            self._hibernate_cond = aio.Condition()
        for module in self._next_module():
            if self.no_more_data() and self.all_blocked() and \
               self.is_waiting_for_input():
                if not self._keep_running:
                    async with self._hibernate_cond:
                        await self._hibernate_cond.wait()
            if self._keep_running:
                self._keep_running -= 1
            if not self._consider_module(module):
                logger.info("Module %s not scheduled"
                            " because of interactive mode",
                            module.name)
                continue
            if not(module.is_ready() or self.has_input()):
                logger.info("Module %s not scheduled"
                            " because not ready and has no input",
                            module.name)
                continue

            self._run_number += 1
            await self._run_tick_procs()
            module.run(self._run_number)
            await module.after_run(self._run_number)
            await aio.sleep(0)

    def _next_module(self):
        """
        Generator the yields a possibly infinite sequence of modules.
        Handles order recomputation and starting logic if needed.
        """
        self._run_index = 0
        first_run = self._run_number
        input_mode = self.has_input()
        self._start_inter = 0
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
                if input_mode:  # end input mode
                    logger.info('Ending interactive mode after %s s',
                                default_timer()-self._start_inter)
                    self._start_inter = 0
                    input_mode = False
                else:
                    self._start_inter = default_timer()
                    logger.info('Starting interactive mode at %s',
                                self._start_inter)
                    input_mode = True
                # Restart from beginning
                self._run_index = 0
                first_run = self._run_number
            module = self._run_list[self._run_index]
            self._run_index += 1  # allow it to be reset
            yield module
            if self._run_index >= len(self._run_list):  # end of modules
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
        self._new_runorder = dataflow.order_modules()  # raises if invalid
        self._new_modules = dataflow.modules()
        self._new_dependencies = dataflow.inputs
        dataflow._compute_reachability(self._new_dependencies)
        self._new_reachability = dataflow.reachability
        self.dataflow.committed()
        self.version += 1  # only increment if valid
        # The slots in the module,_modules, and _runorder will be updated
        # in _update_modules when the scheduler decides it is time to do so.
        if not self._running:  # no need to delay updating the scheduler
            self._update_modules()

    def _update_modules(self):
        if not self._new_modules:
            return
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
        if not(deleted or added):
            logger.info("Scheduler updated with no new module(s)")
        self._dependencies = self._new_dependencies
        self._new_dependencies = None
        self._reachability = self._new_reachability
        self._new_reachability = None
        logger.info("New dependencies: %s", self._dependencies)
        for mid, slots in self._dependencies.items():
            modules[mid].reconnect(slots)
        if added:
            logger.info("Scheduler adding modules %s", added)
            for mid in added:
                modules[mid].starting()
        self._new_modules = None
        self._run_list = []
        self._runorder = self._new_runorder
        self._new_runorder = None
        logger.info("New modules order: %s", self._runorder)
        for i, mid in enumerate(self._runorder):
            module = self._modules[mid]
            self._run_list.append(module)
            module.order = i

    def _end_of_modules(self, first_run):
        # Reset interaction mode
        #self._proc_interaction_opts()
        self._selection_target_time = -1
        new_list = [m for m in self._run_list if not m.is_terminated()]
        self._run_list = new_list
        if first_run == self._run_number:  # no module ready
            has_run = False
            for proc in self._idle_procs:
                # pylint: disable=broad-except
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

    async def idle_proc_runner(self):
        has_run = False
        for proc in self._idle_procs:
            # pylint: disable=broad-except
            try:
                logger.debug('Running idle proc')
                if aio.iscoroutinefunction(proc):
                    await proc(self, self._run_number)
                else:
                    proc(self, self._run_number)
                has_run = True
            except Exception as exc:
                logger.error(exc)
        if not has_run:
            logger.info('sleeping %f', 0.2)
            await aio.sleep(0.2)

    async def _run_tick_procs(self):
        # pylint: disable=broad-except
        for proc in self._tick_procs:
            logger.debug('Calling tick_proc')
            try:
                if aio.iscoroutinefunction(proc):
                    await proc(self, self._run_number)
                else:
                    proc(self, self._run_number)
            except Exception as exc:
                logger.warning(exc)
        for proc in self._tick_once_procs:
            try:
                if aio.iscoroutinefunction(proc):
                    await proc()
                else:
                    proc(self, self._run_number)
            except Exception as exc:
                logger.warning(exc)
            self._tick_once_procs = []

    def exit_(self, *args, **kwargs):
        self._exit = True
        self.resume()
        # self._stopped_evt.set()
        # after the previous statement both _stopped_evt and _not_stopped_evt
        # are set. The goal is allow all tasks to exit

    async def stop(self):
        "Stop the execution."
        async with self._hibernate_cond:
            self._keep_running = KEEP_RUNNING
            self._hibernate_cond.notify()
            self._stopped = True

            
    def task_stop(self):
        return aio.create_task(self.stop())

    def is_running(self):
        "Return True if the scheduler is currently running."
        return self._running

    def is_stopped(self):
        "Return True if the scheduler is stopped."
        return self._stopped

    def is_terminated(self):
        "Return True if the scheduler is terminated."
        for module in self.modules().values():
            if not module.is_terminated():
                return False
        return True

    def done(self):
        self._task = None
        logger.info('Task finished')

    def __len__(self):
        return len(self._modules)

    def exists(self, moduleid):
        "Return True if the moduleid exists in this scheduler."
        return moduleid in self

    def modules(self):
        "Return the dictionary of modules."
        return self._modules

    def __getitem__(self, mid):
        if self.dataflow:
            return self.dataflow[mid]
        return self._modules.get(mid, None)

    def __delitem__(self, name):
        if self.dataflow:
            del self.dataflow[name]
        else:
            raise ProgressiveError('Cannot delete module %s'
                                   'outside a context' % name)

    def __contains__(self, name):
        if self.dataflow:
            return name in self.dataflow
        return name in self._modules

    def run_number(self):
        "Return the last run number."
        return self._run_number

    async def for_input(self, module):
        """
        Notify this scheduler that the module has received input
        that should be served fast.
        """
        async with self._hibernate_cond:
            self._keep_running = KEEP_RUNNING
            self._hibernate_cond.notify()
        sel = self._reachability.get(module.name, False)
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
        self.shortcut_evt.set()
        return self.run_number()+1

    def has_input(self):
        "Return True of the scheduler is in input mode"
        if self._module_selection is None:
            return False
        if not self._module_selection:  # empty, cleanup
            logger.info('Finishing input management')
            self._module_selection = None
            self._selection_target_time = -1
            return False
        return True

    def _consider_module(self, module):
        # FIxME For now, accept all modules in input management
        if not self.has_input():
            return True
        if module.name in self._module_selection:
            # self._module_selection.remove(module.name)
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


    @staticmethod
    def _module_order(x, y):
        if 'order' in x:
            if 'order' in y:
                return x['order']-y['order']
            return 1
        if 'order' in y:
            return -1
        return 0


if Scheduler.default is None:
    Scheduler.default = Scheduler()
