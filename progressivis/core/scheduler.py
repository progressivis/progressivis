"""
Base Scheduler class, runs progressive modules.
"""
from __future__ import annotations

import logging
import functools
from timeit import default_timer
import traceback

from .dataflow import Dataflow
from . import aio
from ..utils.errors import ProgressiveError


from typing import (
    Optional,
    Dict,
    List,
    Sequence,
    Any,
    Callable,
    Set,
    Coroutine,
    Union,
    AsyncGenerator,
    cast,
    TYPE_CHECKING,
)

logger = logging.getLogger(__name__)

__all__ = ["Scheduler"]

KEEP_RUNNING = 5
SHORTCUT_TIME: float = 1.5

if TYPE_CHECKING:
    from progressivis.core.module import Module
    from progressivis.core.slot import Slot

    Dependencies = Dict[str, Dict[str, Slot]]

TickCb = Callable[["Scheduler", int], None]
TickCoro = Callable[["Scheduler", int], Coroutine[Any, Any, Any]]
TickProc = Union[TickCb, TickCoro]
ChangeProc = Callable[
    ["Scheduler", Set["Module"], Set["Module"]], Coroutine[Any, Any, None]
]
Order = List[str]
Reachability = Dict[str, List[str]]


class CallbackList(Dict[TickProc, int]):
    async def fire(self, scheduler: Scheduler, run_number: int) -> bool:
        ret = False
        for proc, count in list(self.items()):
            try:
                if count > 0:
                    count -= 1
                    self[proc] = count
                    continue
                if count == 0:
                    del self[proc]
                if aio.iscoroutinefunction(proc):
                    coro = cast(TickCoro, proc)
                    await coro(scheduler, run_number)
                else:
                    proc(scheduler, run_number)
                ret = True
            except Exception as exc:
                logger.warning(exc)
        return ret


class Scheduler:
    "Base Scheduler class, runs progressive modules"
    # pylint: disable=too-many-public-methods,too-many-instance-attributes
    default: "Scheduler"
    _last_id: int = 0

    @classmethod
    def or_default(cls, scheduler: Optional["Scheduler"]) -> "Scheduler":
        "Return the specified scheduler of, in None, the default one."
        return scheduler or cls.default

    def __init__(self, interaction_latency: int = 1):
        if interaction_latency <= 0:
            raise ProgressiveError(
                "Invalid interaction_latency, should be strictly positive: %s"
                % interaction_latency
            )

        # same as clear below
        Scheduler._last_id += 1
        self._name: int = Scheduler._last_id
        self._modules: Dict[str, Module] = {}
        self._dependencies: Dependencies
        self._running: bool = False
        self._stopped: bool = True
        self._runorder: Order = []
        self._added_modules: Set[Module] = set()
        self._deleted_modules: Set[Module] = set()
        self._start: float = 0
        self._step_once = False
        self._run_number = 0
        self._tick_procs = CallbackList()
        self._idle_procs = CallbackList()
        self._loop_procs = CallbackList()
        self._change_procs: Set[ChangeProc] = set()
        self.version = 0
        self._run_list: List[Module] = []
        self._run_index = 0
        self._module_selection: Optional[Set[str]] = None
        self._selection_target_time: float = -1
        self.interaction_latency = interaction_latency
        self._reachability: Reachability = {}
        self._start_inter: float = 0
        self._hibernate_cond: aio.Condition
        self._keep_running: float = KEEP_RUNNING
        self.dataflow: Optional[Dataflow] = Dataflow(self)
        self.module_iterator = None
        self._enter_cnt = 1
        self._lock: aio.Lock
        self._task = False
        self.shortcut_evt: aio.Event
        self.coros: List[Coroutine[Any, Any, Any]] = []
        self._multiple_slots_name_generator = 1

    def new_run_number(self) -> int:
        self._run_number += 1
        return self._run_number

    def __enter__(self) -> Dataflow:
        if self.dataflow is None:
            self.dataflow = Dataflow(self)
            self.dataflow.multiple_slots_name_generator = (
                self._multiple_slots_name_generator
            )
            self._enter_cnt = 1
        else:
            self._enter_cnt += 1
        return self.dataflow

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        self._enter_cnt -= 1
        if exc_type is not None:
            logger.error("Aborting Dataflow with exception %s", exc_type)
            print(f"Aborting Dataflow with exception {exc_type}")
            traceback.print_tb(tb)
            if self._enter_cnt == 0:
                if self.dataflow is not None:
                    self.dataflow.aborted()
                self.dataflow = None
        if self._enter_cnt > 0 or self.dataflow is None:
            return
        errors = self.dataflow.validate()
        if errors:
            logger.error("Dataflow has errors %s", errors)
            if self.dataflow is not None:
                self.dataflow.aborted()
                self.dataflow = None
            raise ProgressiveError(f"Invalid dataflow: {errors}")

    @property
    def name(self) -> str:
        "Return the scheduler id"
        return str(self._name)

    def timer(self) -> float:
        "Return the scheduler timer."
        if self._start == 0:
            self._start = default_timer()
            return 0
        return default_timer() - self._start

    def run_queue_length(self) -> int:
        "Return the length of the run queue"
        return len(self._run_list)

    def to_json(self, short: bool = True) -> Dict[str, Any]:
        "Return a dictionary describing the scheduler"
        msg: Dict[str, Any] = {}
        mods = {}
        for name, module in self.modules().items():
            mods[name] = module.to_json(short=short)
        modules = sorted(mods.values(), key=functools.cmp_to_key(self._module_order))
        msg["modules"] = modules
        msg["is_running"] = self.is_running()
        msg["is_terminated"] = self.is_terminated()
        msg["run_number"] = self.run_number()
        msg["status"] = "success"
        return msg

    def _repr_html_(self) -> str:
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
        html_head += f"""
<p><b>Scheduler</b> {hex(id(self))}
        <b>{"running" if self.is_running() else "stopped"}</b>,
        <b>modules:</b> {len(self)},
        <b>run number:</b> {self.run_number()}
</p>"""
        html_head += """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Id</th><th>Class</th><th>State</th><th>Last Update</th><th>Order</th>
    </tr>
  </thead>
  <tbody>"""
        columns = ["id", "classname", "state", "last_update", "order"]
        for mod in self._run_list:
            values = mod.to_json(short=True)
            html_head += "<tr>"
            html_head += "".join(
                ["<td>%s</td>" % (values[column]) for column in columns]
            )
        html_end = "</tbody></table>"
        return html_head + html_end

    @staticmethod
    def set_default() -> None:
        "Set the default scheduler."
        if not isinstance(Scheduler.default, Scheduler):
            Scheduler.default = Scheduler()

    def _before_run(self) -> None:
        logger.debug("Before run %d", self._run_number)

    def _after_run(self) -> None:
        pass

    async def start_impl(
        self,
        tick_proc: Optional[TickProc] = None,
        idle_proc: Optional[TickProc] = None,
        coros: Sequence[Coroutine[Any, Any, Any]] = (),
    ) -> None:
        async with self._lock:
            if self._task:
                raise ProgressiveError(
                    "Trying to start scheduler task inside scheduler task"
                )
            print("Starting scheduler")
            self._task = True
        self.coros = list(coros)
        if tick_proc:
            self.on_tick(tick_proc)
        if idle_proc:
            self.on_idle(idle_proc)
        await self.run()

    async def start(
        self,
        tick_proc: Optional[TickProc] = None,
        idle_proc: Optional[TickProc] = None,
        coros: Sequence[Coroutine[Any, Any, Any]] = (),
        persist: bool = False,
    ) -> None:
        from ..storage import init_temp_dir_if, cleanup_temp_dir, temp_dir

        if self._task:
            return
        self.shortcut_evt = aio.Event()
        self._hibernate_cond = aio.Condition()
        self._lock = aio.Lock()
        itd_flag = False
        if not persist:
            return await self.start_impl(tick_proc, idle_proc, coros)
        try:
            itd_flag = init_temp_dir_if()
            if itd_flag:
                print("Init TEMP_DIR in start()", temp_dir())
            return await self.start_impl(tick_proc, idle_proc, coros)
        finally:
            if itd_flag:
                cleanup_temp_dir()

    def task_start(self, *args: Any, **kwargs: Any) -> aio.Task[Any]:
        return aio.create_task(self.start(*args, **kwargs))

    def _step_proc(self, s: Scheduler, run_number: int) -> None:
        # pylint: disable=unused-argument
        self.task_stop()

    async def step(self) -> None:
        "Start the scheduler for on step."
        await self.start(tick_proc=self._step_proc)

    def on_tick(self, proc: TickProc, delay: int = -1) -> None:
        "Set a procedure to call at each tick."
        assert callable(proc)
        self._tick_procs[proc] = delay

    def remove_tick(self, proc: TickProc) -> None:
        "Remove a tick callback"
        self._tick_procs.pop(proc, None)

    def on_tick_once(self, proc: TickProc) -> None:
        """
        Add a oneshot function that will be run at the next scheduler tick.
        This is especially useful for setting up module connections.
        """
        self.on_tick(proc, 1)

    def remove_tick_once(self, proc: TickProc) -> None:
        "Remove a tick once callback"
        self.remove_tick(proc)

    def on_idle(self, proc: TickProc, delay: int = -1) -> None:
        "Set a procedure that will be called when there is nothing else to do."
        assert callable(proc)
        self._idle_procs[proc] = delay

    def remove_idle(self, idle_proc: TickProc) -> None:
        "Remove an idle callback."
        self._idle_procs.pop(idle_proc, None)

    def on_loop(self, proc: TickProc, delay: int = -1) -> None:
        assert callable(proc)
        self._loop_procs[proc] = delay

    def remove_loop(self, idle_proc: TickProc) -> None:
        "Remove an idle callback."
        self._loop_procs.pop(idle_proc, None)

    def on_change(self, proc: ChangeProc) -> None:
        assert callable(proc)
        self._change_procs.add(proc)

    def remove_change(self, proc: ChangeProc) -> None:
        self._change_procs.remove(proc)

    async def run(self) -> None:
        "Run the modules, called by start()."
        global KEEP_RUNNING
        self.commit()
        self._stopped = False
        self._running = True
        self._start = default_timer()
        self._before_run()
        run_loop = aio.create_task(self._run_loop())
        shortcut_loop = aio.create_task(self.shortcut_manager())
        runners = [run_loop, shortcut_loop]
        runners.extend([aio.create_task(coro) for coro in self.coros])
        for r in runners:
            assert isinstance(r, aio.Task)
        # TODO: find the "right" initialisation value ...
        KEEP_RUNNING = min(50, len(self._run_list) * 3)
        self._keep_running = KEEP_RUNNING
        await aio.gather(*runners)
        self._running = False

        self._stopped = True
        self._task = False
        self._after_run()
        self.done()

    async def shortcut_manager(self) -> None:
        while not self._stopped and self._run_list:
            await self.shortcut_evt.wait()
            if self._stopped or not self._run_list:
                break
            await aio.sleep(SHORTCUT_TIME)
            self._module_selection = None
            self.shortcut_evt.clear()

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        # pylint: disable=broad-except
        blocked = 0  # all_blocked() cannot detect that all modules are blocked
        async for module in self._next_module():
            await aio.sleep(0)
            if (
                self.no_more_data()
                and (self.all_blocked() or blocked == len(self._run_list))
                and self.is_waiting_for_input()
            ):
                if self._keep_running <= 0:
                    async with self._hibernate_cond:
                        await self._hibernate_cond.wait()
            if self._keep_running > 0:
                self._keep_running -= 1
            if not self._consider_module(module):
                logger.info(
                    "Module %s not scheduled because of interactive mode",
                    module.name,
                )
                continue
            # increment the run number, even if we don't call the module
            self._run_number += 1
            module.prepare_run(self._run_number)
            if not (module.is_ready() or self.has_input() or module.is_greedy()):
                logger.info(
                    "Module %s not scheduled because not ready and has no input",
                    module.name,
                )
                blocked += 1
                continue
            blocked = 0
            await module.start_run(self._run_number)
            module.run(self._run_number)
            await module.after_run(self._run_number)
            await self._run_tick_procs()
        if self.shortcut_evt is not None:
            self.shortcut_evt.set()
        print("Leaving run loop")

    async def _next_module(self) -> AsyncGenerator[Module, None]:
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
            if self.dataflow is not None and self._enter_cnt == 0:
                self._update_modules()
                self._run_index = 0
                first_run = self._run_number
            if self._deleted_modules:
                for mod in self._deleted_modules:
                    await mod.ending()
            if self._added_modules or self._deleted_modules:
                added = self._added_modules
                deleted = self._deleted_modules
                self._added_modules = set()
                self._deleted_modules = set()
                for proc in self._change_procs:
                    await proc(self, added, deleted)
            # If run_list empty, we're done
            if not self._run_list:
                break
            # Check for interactive input mode
            if input_mode != self.has_input():
                if input_mode:  # end input mode
                    logger.info(
                        "Ending interactive mode after %s s",
                        default_timer() - self._start_inter,
                    )
                    self._start_inter = 0
                    input_mode = False
                else:
                    self._start_inter = default_timer()
                    logger.info("Starting interactive mode at %s", self._start_inter)
                    input_mode = True
                # Restart from beginning
                self._run_index = 0
                first_run = self._run_number
            module = self._run_list[self._run_index]
            self._run_index += 1  # allow it to be reset
            yield module
            if self._run_index >= len(self._run_list):  # end of modules
                await self._end_of_modules(first_run)
                first_run = self._run_number

    def all_blocked(self) -> bool:
        "Return True if all the modules are blocked, False otherwise"
        from .module import Module

        for module in self._run_list:
            if module.state not in (Module.state_blocked, Module.state_suspended):
                # print("all_blocked: False")
                return False
        # print("all_blocked: True")
        return True

    def is_waiting_for_input(self) -> bool:
        "Return True if there is at least one input module"
        for module in self._run_list:
            if module.is_input():
                # print("is_waiting_for_input: True")
                return True
        # print("is_waiting_for_input: False")
        return False

    def no_more_data(self) -> bool:
        "Return True if at least one module has data input."
        for module in self._run_list:
            if module.is_data_input():
                # print("no_more_data: False")
                return False
        # print("no_more_data: True")
        return True

    def commit(self) -> None:
        """Forces a pending dataflow to be commited

        :returns: None

        """
        if self.dataflow is None or self._enter_cnt > 1:
            return
        self._enter_cnt = 0
        if not self._running:  # no need to delay updating the scheduler
            self._update_modules()

    def _update_modules(self) -> None:
        if self.dataflow is None or self._enter_cnt != 0:
            return
        dataflow = self.dataflow
        _new_runorder = dataflow.order_modules()  # raises if invalid
        _new_modules = list(dataflow.modules().values())
        _new_inputs = dataflow.inputs
        _new_outputs = dataflow.outputs
        dataflow._compute_reachability(_new_inputs)
        _new_reachability = dataflow.reachability
        dataflow.committed()
        self._multiple_slots_name_generator = (
            self.dataflow.multiple_slots_name_generator
        )
        self.dataflow = None
        logger.info("Updating modules")
        prev_keys = set(self._modules.keys())
        modules = {module.name: module for module in _new_modules}
        keys = set(modules.keys())
        added = keys - prev_keys
        deleted = prev_keys - keys
        if deleted:
            logger.info(f"Scheduler deleted modules: {deleted}")
            print(f"# Scheduler deleted module(s): {deleted}")
            self._deleted_modules.update({self[mid] for mid in deleted})
        self._modules = modules
        if not (deleted or added):
            logger.info("Scheduler updated with no new module(s)")
        self._dependencies = _new_inputs
        self._reachability = _new_reachability
        logger.info("New dependencies: %s", self._dependencies)
        for mid, slots in self._dependencies.items():
            modules[mid].reconnect(slots, _new_outputs.get(mid, {}))
        _new_outputs = {}
        _new_inputs = {}
        _new_reachability = {}
        if added:
            logger.info("Scheduler adding modules %s", added)
            sorted_added = sorted(added)
            print(f"# Scheduler added module(s): {sorted_added}")
            for mid in added:
                modules[mid].starting()
            self._added_modules.update({self[mid] for mid in added})
        self._run_list = []
        self._runorder = _new_runorder
        logger.info("New module order: %s", self._runorder)
        for i, mid in enumerate(self._runorder):
            module = self._modules[mid]
            self._run_list.append(module)
            module.order = i
        if not self._run_list:
            print("# Scheduler empty, finishing")

    async def _end_of_modules(self, first_run: int) -> None:
        # Reset interaction mode
        self._selection_target_time = -1
        new_list = [m for m in self._run_list if not m.is_terminated()]
        self._run_list = new_list
        await self._loop_procs.fire(self, self._run_number)
        if first_run == self._run_number:  # no module ready
            has_run = await self._idle_procs.fire(self, self._run_number)
            if not has_run:
                logger.info("sleeping %f", 0.2)
                print("Sleeping 0.2")
                aio.sleep(0.2)
        self._run_index = 0

    async def idle_proc_runner(self) -> None:
        has_run = False
        for proc in self._idle_procs:
            # pylint: disable=broad-except
            try:
                logger.debug("Running idle proc")
                if aio.iscoroutinefunction(proc):
                    coro = cast(TickCoro, proc)
                    await coro(self, self._run_number)
                else:
                    proc(self, self._run_number)
                has_run = True
            except Exception as exc:
                logger.error(exc)
        if not has_run:
            logger.info("sleeping %f", 0.2)
            await aio.sleep(0.2)

    async def _run_tick_procs(self) -> None:
        # pylint: disable=broad-except
        await self._tick_procs.fire(self, self._run_number)

    async def stop(self) -> None:
        "Stop the execution."
        self._stopped = True
        if self.shortcut_evt is not None:
            self.shortcut_evt.set()
        async with self._hibernate_cond:
            self._keep_running = KEEP_RUNNING
            self._hibernate_cond.notify()
            self._stopped = True

    def task_stop(self) -> Optional[aio.Task[Any]]:
        if self.is_running():
            return aio.create_task(self.stop())
        return None

    def is_running(self) -> bool:
        "Return True if the scheduler is currently running."
        return self._running

    def is_stopped(self) -> bool:
        "Return True if the scheduler is stopped."
        return self._stopped

    def is_terminated(self) -> bool:
        "Return True if the scheduler is terminated."
        for module in self.modules().values():
            if not module.is_terminated():
                return False
        return True

    def done(self) -> None:
        self._task = False
        logger.info("Task finished")

    def __len__(self) -> int:
        return len(self.modules())

    def exists(self, moduleid: str) -> bool:
        "Return True if the moduleid exists in this scheduler."
        return moduleid in self

    def modules(self) -> Dict[str, Module]:
        "Return the dictionary of modules."
        return self._modules

    def __getitem__(self, mid: str) -> Module:
        if self.dataflow is not None:
            return self.dataflow[mid]
        return self._modules[mid]

    def __delitem__(self, name: str) -> None:
        if self.dataflow is not None:
            self.dataflow.delete_modules(name)
        else:
            raise ProgressiveError("Cannot delete module %soutside a context" % name)

    def __contains__(self, name: str) -> bool:
        if self.dataflow is not None:
            return name in self.dataflow
        return name in self._modules

    def groups(self) -> Set[str]:
        return {mod.group for mod in self.modules().values() if mod.group is not None}

    def group_modules(self, *names: str) -> List[str]:
        nameset = set(names)
        if not nameset:
            return []
        return [mod.name for mod in self.modules().values() if mod.group in nameset]

    def run_number(self) -> int:
        "Return the last run number."
        return self._run_number

    def _ipython_key_completions_(self) -> List[str]:
        return list(self._modules.keys())

    async def wake_up(self):
        async with self._hibernate_cond:
            self._keep_running = KEEP_RUNNING
            self._hibernate_cond.notify()

    async def for_input(self, module: Module) -> int:
        """
        Notify this scheduler that the module has received input
        that should be served fast.
        """
        await self.wake_up()
        sel = self._reachability.get(module.name, None)
        if sel:
            if not self._module_selection:
                logger.info("Starting input management")
                self._module_selection = set(sel)
                self._selection_target_time = self.timer() + self.interaction_latency
            else:
                self._module_selection.update(sel)
            logger.debug("Input selection for module: %s", self._module_selection)
        self.shortcut_evt.set()
        return self.run_number() + 1

    def has_input(self) -> bool:
        "Return True of the scheduler is in input mode"
        if self._module_selection is None:
            return False
        if not self._module_selection:  # empty, cleanup
            logger.info("Finishing input management")
            self._module_selection = None
            self._selection_target_time = -1
            return False
        return True

    def _consider_module(self, module: Module) -> bool:
        # FIxME For now, accept all modules in input management
        if not self.has_input():
            return True
        if self._module_selection and module.name in self._module_selection:
            # self._module_selection.remove(module.name)
            logger.debug("Module %s ready for scheduling", module.name)
            return True
        logger.debug("Module %s NOT ready for scheduling", module.name)
        return False

    def time_left(self) -> float:
        "Return the time left to run for this slot."
        if self._selection_target_time <= 0 and not self.has_input():
            logger.error("time_left called with no target time")
            return 0
        return max(0, self._selection_target_time - self.timer())

    def fix_quantum(self, module: Module, quantum: float) -> float:
        "Fix the quantum of the specified module"
        if (
            self.has_input()
            and self._module_selection  # redundant
            and module.name in self._module_selection
        ):
            quantum = self.time_left() / len(self._module_selection)
        if quantum == 0:
            quantum = 0.1
            logger.info(
                "Quantum is 0 in %s, setting it to a reasonable value", module.name
            )
        return quantum

    def close_all(self) -> None:
        "Close all the resources associated with this scheduler."
        for mod in self.modules().values():
            mod.close_all()

    @staticmethod
    def _module_order(x: Dict[str, int], y: Dict[str, int]) -> int:
        if "order" in x:
            if "order" in y:
                return x["order"] - y["order"]
            return 1
        if "order" in y:
            return -1
        return 0


Scheduler.default = Scheduler()
