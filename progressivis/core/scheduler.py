"""
Base Scheduler class, runs progressive modules.
"""
import logging
import functools
from timeit import default_timer

from .dataflow import Dataflow
import progressivis.core.aio as aio
from progressivis.utils.errors import ProgressiveError

logger = logging.getLogger(__name__)

__all__ = ['Scheduler']

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
        #with self.lock:
        Scheduler._last_id += 1
        self._name = Scheduler._last_id
        self._modules = {}
        self._dependencies = None
        self._running = False
        self._stopped = True
        self._stopped_evt = None
        self._not_stopped_evt = None
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
        self._inter_cycles_cnt = 0
        self._interaction_opts = None
        self.dataflow = Dataflow(self)
        self.module_iterator = None
        self._enter_cnt = 1
        self.runners = set()
        self.coros = []
        self._short_cycle = False


    def new_run_number(self):
        self._run_number += 1
        return self._run_number

    def join(self):
        "Wait for this execution thread to finish."

    @property
    def lock(self):
        "Return the scheduler lock."
        return self._lock

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
        #with self.lock:
        for (name, module) in self.modules().items():
            mods[name] = module.to_json(short=short)
        modules = sorted(mods.values(), key=functools.cmp_to_key(self._module_order))
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

    async def unlocker(self):
        while True:
            await aio.sleep(0.5)
            new_list = []
            for m in self._run_list:
                m.is_ready()
                if m.is_terminated():
                    m.notify_consumers()
                    self.runners.remove(m.name)
                    m.steering_evt_set()
                    continue
                new_list.append(m)
            self._run_list = new_list
            if self._short_cycle:
                mods_selection = [m for m in self._run_list if m.name in self._module_selection]
            else:
                mods_selection = self._run_list
            if not len(mods_selection):
                self.exit()
                break
            nb_waiters = sum([m._w8_slots for m in mods_selection])
            nb_max_w8 = len([x for x in mods_selection if not x.is_source() and x.has_any_input()])
            if nb_waiters >= nb_max_w8-1 or self._stopped or self._short_cycle:
                if self._idle_procs:
                    await self.idle_proc_runner()
                self.unfreeze()
                for m in self._run_list:
                    m.steering_evt_set()
            if self._exit:
                break
            await self._not_stopped_evt.wait()
                      
    async def start(self, tick_proc=None, idle_proc=None, coros=()):
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
        #from .sentinel import Sentinel
        #import pdb;pdb.set_trace()
        #sl = Sentinel(scheduler=self)        
        if self.dataflow:
            assert self._enter_cnt == 1
            self._commit(self.dataflow)
            self.dataflow = None
            self._enter_cnt = 0
        self._stopped = False
        self._stopped_evt = aio.Event()        
        self._not_stopped_evt = aio.Event()
        self._not_stopped_evt.set()
        self._running = True
        self._start = default_timer()
        self._before_run()
        if self._new_modules:
            self._update_modules()
        # currently, the order in self._run_list is not important anymore
        runners = [aio.create_task(m.module_task(), m.name)
                   for m in self._run_list]
        runners.extend([aio.create_task(coro)
                        for coro in self.coros])
        runners.append(aio.create_task(self.unlocker(), "unlocker"))
        self.runners = [m.name for m in self._run_list]
        await aio.gather(*runners)
        modules = [self._modules[m] for m in self._runorder]
        for module in reversed(modules):
            module.ending()
        self._running = False
        self._stopped = True
        self._after_run()
        self.done()


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
        #with self.lock:
        self._run_list = []
        self._runorder = self._new_runorder            
        self._new_runorder = None
        logger.info("New modules order: %s", self._runorder)
        for i, mid in enumerate(self._runorder):
            module = self._modules[mid]
            self._run_list.append(module)
            module.order = i


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
        #import pdb;pdb.set_trace()
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


    def exit(self, *args, **kwargs):
        self._exit = True
        self.resume()
        self._stopped_evt.set()
        # after the previous statement both _stopped_evt and _not_stopped_evt
        # are set. The goal is allow all tasks to exit
        for m in self._run_list:
            m.steering_evt_set()

    def stop(self):
        "Stop the execution."
        self._stopped = True
        self._stopped_evt.set()
        self._not_stopped_evt.clear()

    def resume(self, tick_proc=None):
        "Resume the execution."
        self._stopped = False
        self._stopped_evt.clear()
        self._not_stopped_evt.set()
        self._step_once = False
        if tick_proc:
            assert callable(tick_proc)
            self._tick_procs = [tick_proc]
        else:
            self._tick_procs = []


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
        "Called when the execution is done. Can be overridden in subclasses."
        pass

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


    def for_input(self, module):
        """
        Notify this scheduler that the module has received input
        that should be served fast.
        """
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

    def freeze(self):
        self.unfreeze()
        mods_to_freeze = set([m for m in self._run_list if m.name not in self._module_selection and not m.is_input()])
        for module in mods_to_freeze:
            module._frozen = True
            module.steering_evt_clear()
        self._short_cycle = True

    def unfreeze(self):
        if not self._short_cycle:
            return
        for module in self._run_list:
            #if not module._frozen: continue
            module._frozen = False
            module.steering_evt_set()
            #module.release()
        self._short_cycle = False

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
