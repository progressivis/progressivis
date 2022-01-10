"""
Base class for progressive modules.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from traceback import print_exc
import re
import logging
from enum import IntEnum
import pdb

import numpy as np
from progressivis.utils.errors import ProgressiveError, ProgressiveStopIteration
from progressivis.table.table_base import BaseTable
from progressivis.table.table import Table
from progressivis.table.dshape import dshape_from_dtype
from progressivis.table.row import Row
from progressivis.storage import Group
import progressivis.core.aio as aio


from .utils import type_fullname, get_random_name
from .slot import SlotDescriptor, Slot
from .tracer_base import Tracer
from .time_predictor import TimePredictor
from .storagemanager import StorageManager
from .scheduler import Scheduler


from typing import (
    cast,
    Any,
    Iterable,
    Sequence,
    Optional,
    Sized,
    Dict,
    Set,
    List,
    Tuple,
    Callable,
    Type,
    Union,
    ClassVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .dataflow import Dataflow
    from .decorators import _Context

    Parameters = List[Tuple[str, np.dtype[Any], Any]]
JSon = Dict[str, Any]
# ReturnRunStep = Tuple[int, ModuleState]
ReturnRunStep = Dict[str, int]


logger = logging.getLogger(__name__)


class ModuleMeta(ABCMeta):
    """Module metaclass is needed to collect the input parameter list
    in the field ``all_parameters''.
    """

    def __init__(cls, name: str, bases: Any, attrs: Dict[str, Any]) -> None:
        if "parameters" not in attrs:
            cls.parameters: List[Parameters] = []
        if "inputs" not in attrs:
            cls.inputs: List[SlotDescriptor] = []
        if "outputs" not in attrs:
            cls.outputs: List[SlotDescriptor] = []
        all_parameters = list(cls.parameters)
        all_inputs = list(cls.inputs)
        all_outputs = {c.name: c for c in cls.outputs}
        for base in bases:
            all_parameters += getattr(base, "all_parameters", [])
            all_inputs += getattr(base, "all_inputs", [])
            for outp in getattr(base, "all_outputs", []):
                assert isinstance(outp, SlotDescriptor)
                if outp.name not in all_outputs:
                    all_outputs[outp.name] = outp
        cls.all_parameters = all_parameters
        cls.all_inputs = all_inputs
        cls.all_outputs = list(all_outputs.values())
        super(ModuleMeta, cls).__init__(name, bases, attrs)


class ModuleTag:
    tags: Set[str] = set()

    def __init__(self, *tag_list: str):
        self._saved = ModuleTag.tags
        ModuleTag.tags = set(tag_list)

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> Any:
        ModuleTag.tags = self._saved
        return False


class ModuleState(IntEnum):
    state_created = 0
    state_ready = 1
    state_running = 2
    state_blocked = 3
    state_zombie = 4
    state_terminated = 5
    state_invalid = 6


class Module(metaclass=ModuleMeta):
    """The Module class is the base class for all the progressive modules.
    """

    parameters: Parameters = [
        ("quantum", np.dtype(float), 0.5),
        ("debug", np.dtype(bool), False),
    ]
    all_parameters: ClassVar[Parameters]  # defined by metaclass, declare for mypy
    TRACE_SLOT = "_trace"
    PARAMETERS_SLOT = "_params"

    TAG_VISUALIZATION = "visualization"
    TAG_INPUT = "input"
    TAG_SOURCE = "source"
    TAG_GREEDY = "greedy"
    TAG_DEPENDENT = "dependent"

    inputs = [SlotDescriptor(PARAMETERS_SLOT, type=BaseTable, required=False)]
    outputs = [SlotDescriptor(TRACE_SLOT, type=BaseTable, required=False)]
    all_inputs: ClassVar[List[SlotDescriptor]]  # defined by metaclass, declare for mypy
    all_outputs: ClassVar[
        List[SlotDescriptor]
    ]  # defined by metaclass, declare for mypy

    state_created: ClassVar[ModuleState] = ModuleState.state_created
    state_ready: ClassVar[ModuleState] = ModuleState.state_ready
    state_running: ClassVar[ModuleState] = ModuleState.state_running
    state_blocked: ClassVar[ModuleState] = ModuleState.state_blocked
    state_zombie: ClassVar[ModuleState] = ModuleState.state_zombie
    state_terminated: ClassVar[ModuleState] = ModuleState.state_terminated
    state_invalid: ClassVar[ModuleState] = ModuleState.state_invalid

    # state_created = 0
    # state_ready = 1
    # state_running = 2
    # state_blocked = 3
    # state_zombie = 4
    # state_terminated = 5
    # state_invalid = 6
    # state_name = [
    #     "created",
    #     "ready",
    #     "running",
    #     "blocked",
    #     "zombie",
    #     "terminated",
    #     "invalid",
    # ]

    def __new__(cls, *args: Tuple[str, Any], **kwds: Any) -> Module:
        module = object.__new__(cls)
        # pylint: disable=protected-access
        module._args = args
        module._kwds = kwds
        return module

    def __init__(
        self,
        name: Optional[str] = None,
        group: Optional[str] = None,
        scheduler: Optional[Scheduler] = None,
        storagegroup: Optional[Group] = None,
        **kwds: Any,
    ) -> None:
        self._args: Sequence[Tuple[str, Any]]
        self._kwds: Dict[str, Any]
        if scheduler is None:
            scheduler = Scheduler.default
        self._scheduler: Scheduler = scheduler
        if scheduler.dataflow is None:
            raise ProgressiveError("No valid context in scheduler")
        dataflow: Dataflow = scheduler.dataflow
        if name is None:
            name = dataflow.generate_name(self.pretty_typename())
        elif name in dataflow:
            raise ProgressiveError(
                "module already exists in scheduler," " delete it first"
            )
        self.name = name  # need to set the name so exception can remove it
        predictor = TimePredictor.default()
        predictor.name = name
        self.predictor = predictor
        storage = StorageManager.default
        self.storage = storage
        if storagegroup is None:
            assert Group.default_internal is not None
            storagegroup = Group.default_internal(get_random_name(name + "_tracer"))
        self.storagegroup: Group = storagegroup
        tracer = Tracer.default(name, storagegroup)

        self.tags = set(ModuleTag.tags)
        self.order = -1
        self.group = group
        self.tracer = tracer
        self._start_time: float = 0
        self._end_time: float = 0
        self._last_update: int = 0
        self._state = Module.state_created
        self._had_error = False
        self._parse_parameters(kwds)

        # always present
        input_descriptors = self.all_inputs
        output_descriptors = self.all_outputs
        self._input_slots: Dict[str, Optional[Slot]] = self._validate_descriptors(
            input_descriptors
        )
        self.input_descriptors: Dict[str, SlotDescriptor] = {
            d.name: d for d in input_descriptors
        }
        self.input_multiple: Dict[str, int] = {
            d.name: 0 for d in input_descriptors if d.multiple
        }
        self._output_slots: Dict[
            str, Optional[List[Slot]]
        ] = self._validate_descriptors(output_descriptors)
        self.output_descriptors: Dict[str, SlotDescriptor] = {
            d.name: d for d in output_descriptors
        }
        self.default_step_size: int = 100
        self.input = InputSlots(self)
        self.output = OutputSlots(self)
        self.steps_acc: int = 0
        self.wait_expr = aio.FIRST_COMPLETED
        self.after_run_proc = None
        self.context: Optional[_Context] = None
        # callbacks
        self._start_run: Optional[Callable[[Module, int], None]] = None
        self._end_run: Optional[Callable[[Module, int], None]] = None
        dataflow.add_module(self)

    @staticmethod
    def tagged(*tags: str) -> ModuleTag:
        """Create a context manager to add tags to a set of modules
        created within a scope, typically dependent modules.
        """
        return ModuleTag(*tags)

    def scheduler(self) -> Scheduler:
        """Return the scheduler associated with the module.
        """
        return self._scheduler

    def dataflow(self) -> Optional[Dataflow]:
        """Return the dataflow associated with the module at creation time.
        """
        return self._scheduler.dataflow

    # def create_dependent_modules(self, *params, **kwds) -> None:  # pragma no cover
    #     """Create modules that this module depends on.
    #     """
    #     pass

    def close_all(self) -> None:
        if self._params is not None and self._params.storagegroup is not None:
            self._params.storagegroup.close_all()

    def get_progress(self) -> Tuple[int, int]:
        """Return a tuple of numbers (current,total) where current is `current`
        progress value and `total` is the total number of values to process;
        these values can change during the computations.
        """
        if not self.has_any_input():
            return (0, 0)
        slots = self.input_slot_values()
        progresses = []
        for slot in slots:
            if slot is not None:
                progresses.append(slot.output_module.get_progress())
        if len(progresses) == 1:
            return progresses[0]
        elif not progresses:
            return (0, 0)
        pos = 0
        size = 0
        for prog in progresses:
            pos += prog[0]
            size += prog[1]
        return (pos, size)

    def get_quality(self) -> float:
        # pylint: disable=no-self-use
        """Quality value, should increase.
        """
        return 0.0

    async def after_run(self, rn: int) -> None:
        if self.after_run_proc is None:
            return
        proc = self.after_run_proc
        if aio.iscoroutinefunction(proc):
            await proc(self, rn)
        else:
            proc(self, rn)

    # @staticmethod
    # def _add_slots(kwds: Dict[str, List[Slot]],
    #                kwd: str,
    #                slots: List[Slot]) -> None:
    #     if kwd in kwds:
    #         kwds[kwd] += slots
    #     else:
    #         kwds[kwd] = slots

    @staticmethod
    def _validate_descriptors(descriptor_list: List[SlotDescriptor]) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        for desc in descriptor_list:
            if desc.name in slots:
                raise ProgressiveError(
                    "Duplicate slot name %s" f" in slot descriptor {desc.name}"
                )
            slots[desc.name] = None
        return slots

    @property
    def debug(self) -> bool:
        "Return the value of the debug property"
        return bool(self.params.debug)

    @debug.setter
    def debug(self, value: bool) -> None:
        """Set the value of the debug property.

        when True, the module trapped into the debugger when the run_step
        method is called.
        """
        # TODO: should change the run_number of the params
        self.params.debug = bool(value)

    def _parse_parameters(self, kwds: Dict[str, Any]) -> None:
        # pylint: disable=no-member
        self._params = _create_table(
            self.generate_table_name("params"), self.all_parameters
        )
        self.params = Row(self._params)
        for (name, _, _) in self.all_parameters:
            if name in kwds:
                self.params[name] = kwds.pop(name)

    def generate_table_name(self, name: str) -> str:
        "Return a uniq name for this module"
        return f"s{self.scheduler().name}_{self.name}_{name}"

    def timer(self) -> float:
        "Return the timer associated with this module"
        return self.scheduler().timer()

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        "Return a dictionary describing the module"
        s = self.scheduler()
        speed_h: List[Optional[float]] = [1.0]
        if with_speed:
            speed_h = self.tracer.get_speed()
        json = {
            "is_running": s.is_running(),
            "is_terminated": s.is_terminated(),
            "run_number": s.run_number(),
            "id": self.name,
            "classname": self.pretty_typename(),
            "is_visualization": self.is_visualization(),
            "last_update": self._last_update,
            "state": self._state.name,
            "quality": self.get_quality(),
            "progress": list(self.get_progress()),
            "speed": speed_h,
        }
        if self.order >= 0:
            json["order"] = self.order

        if short:
            return json

        json.update(
            {
                "start_time": self._start_time,
                "end_time": self._end_time,
                "input_slots": {
                    k: _islot_to_json(s) for (k, s) in self._input_slots.items()
                },
                "output_slots": {
                    l: _oslot_to_json(t) for (l, t) in self._output_slots.items()
                },
                "default_step_size": self.default_step_size,
                "parameters": self.current_params().to_json(),
            }
        )
        return json

    async def from_input(self, msg: JSon) -> str:
        "Catch and process a message from an interaction"
        if "debug" in msg:
            self.debug = bool(msg["debug"])
        return ""

    def is_input(self) -> bool:
        # pylint: disable=no-self-use
        "Return True if this module is an input module"
        return self.TAG_INPUT in self.tags

    def is_data_input(self) -> bool:
        # pylint: disable=no-self-use
        "Return True if this module brings new data"
        return False

    def get_image(self, run_number: Optional[int] = None) -> Any:  # pragma no cover
        "Return an image created by this module or None"
        # pylint: disable=unused-argument, no-self-use
        return None

    def describe(self) -> None:
        "Print the description of this module"
        print("id: %s" % self.name)
        print("class: %s" % type_fullname(self))
        print("quantum: %f" % self.params.quantum)
        print("start_time: %s" % self._start_time)
        print("end_time: %s" % self._end_time)
        print("last_update: %s" % self._last_update)
        print("state: %s(%d)" % (self._state.name, self._state.value))
        print("input_slots: %s" % self._input_slots)
        print("outpus_slots: %s" % self._output_slots)
        print("default_step_size: %d" % self.default_step_size)
        if self._params:
            print("parameters: ")
            print(self._params)

    def pretty_typename(self) -> str:
        "Return a the type name of this module in a pretty form"
        name = self.__class__.__name__
        pretty = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        pretty = re.sub("([a-z0-9])([A-Z])", r"\1_\2", pretty).lower()
        pretty = re.sub("_module$", "", pretty)
        return pretty

    def __str__(self) -> str:
        return "Module %s: %s" % (self.__class__.__name__, self.name)

    def __repr__(self) -> str:
        return str(self)

    async def start(self) -> None:
        "Start the scheduler associated with this module"
        await self.scheduler().start()

    def terminate(self) -> None:
        "Set the state to terminated for this module"
        self.state = Module.state_zombie

    def create_slot(
        self,
        output_name: Union[str, int],
        input_module: Optional[Module],
        input_name: Optional[str],
    ) -> Slot:
        "Create a specified output slot"
        if isinstance(output_name, int):
            pos = output_name
            assert pos == 0  # TODO: try to remove this restriction
            slot_desc = self.all_outputs[pos]
            assert slot_desc.name == "result"  # TODO: Idem
            output_name = slot_desc.name
        assert isinstance(output_name, str)
        return Slot(self, output_name, input_module, input_name)

    def connect_output(
        self, output_name: str, input_module: Module, input_name: str
    ) -> Slot:
        "Connect the output slot"
        slot = self.create_slot(output_name, input_module, input_name)
        slot.connect()
        return slot

    def has_any_input(self) -> bool:
        "Return True if the module has any input"
        return any(self._input_slots.values())

    def has_input_slot(self, name: str) -> bool:
        return self._input_slots.get(name, None) is not None

    def get_input_slot(self, name: str) -> Slot:
        "Return the specified input slot"
        # raises error is the slot is not declared
        slot = self._input_slots[name]
        if slot is None:
            raise KeyError(f"slot '{name}' not connected")
        return slot

    def get_input_slot_multiple(self, name: str) -> List[str]:
        if not self.input_slot_multiple(name):
            return [name]  # self.get_input_slot(name)]
        prefix = name + "."
        return [iname for iname in self._input_slots if iname.startswith(prefix)]

    def get_input_module(self, name: str) -> Optional[Module]:
        "Return the specified input module"
        slot = self.get_input_slot(name)
        assert slot is not None
        return slot.output_module

    def input_slot_values(self) -> List[Slot]:
        return [slot for slot in self._input_slots.values() if slot is not None]

    def input_slot_descriptor(self, name: str) -> SlotDescriptor:
        return self.input_descriptors[name]

    def input_slot_type(self, name: str) -> Any:
        return self.input_descriptors[name].type

    def input_slot_required(self, name: str) -> bool:
        return self.input_descriptors[name].required

    def input_slot_multiple(self, name: str) -> bool:
        return self.input_descriptors[name].multiple

    def input_slot_names(self) -> Iterable[str]:
        return self._input_slots.keys()

    def reconnect(self, inputs: Dict[str, Slot]) -> None:
        deleted_keys = set(self._input_slots.keys()) - set(inputs.keys())
        for name, slot in inputs.items():
            old_slot = self._input_slots.get(name, None)
            if old_slot is not slot:
                # pylint: disable=protected-access
                assert slot.input_module is self
                if slot.original_name:
                    descriptor = self.input_descriptors[slot.original_name]
                    # self.input_descriptors[name] = descriptor
                    self.inputs.append(descriptor)
                    logger.info(
                        'Creating multiple input slot "%s" in "%s"', name, self.name
                    )
                self._input_slots[name] = slot
                if old_slot:
                    old_slot.output_module._disconnect_output(old_slot.output_name)
                slot.output_module._connect_output(slot)

        for name in deleted_keys:
            old_slot = self._input_slots[name]
            if old_slot:
                logger.info(f"Deleted input slot {name} in {self.name}")
                # pylint: disable=protected-access
                old_slot.output_module._disconnect_output(old_slot.output_name)
                if old_slot.original_name:
                    descriptor = self.input_descriptors.pop(name)
                    assert descriptor.name == old_slot.original_name
                    del self.inputs[self.inputs.index(descriptor)]
                    logger.info(
                        'Removing multiple input slot "%s" in "%s"', name, self.name
                    )
            self._input_slots[name] = None

    def has_any_output(self) -> bool:
        return any(self._output_slots.values())

    def get_output_slot(self, name: str) -> Optional[List[Slot]]:
        # raise error is the slot is not declared
        return self._output_slots[name]

    def output_slot_descriptor(self, name: str) -> SlotDescriptor:
        return self.output_descriptors[name]

    def output_slot_type(
        self, name: str
    ) -> Optional[Union[Type[Any], Tuple[Type[Any], ...]]]:
        return self.output_descriptors[name].type

    def output_slot_values(self) -> Iterable[Optional[List[Slot]]]:
        return self._output_slots.values()

    def output_slot_names(self) -> Iterable[str]:
        return self._output_slots.keys()

    def validate(self) -> None:
        "called when the module have been validated"
        if self.state == self.state_created:
            self.state = Module.state_blocked

    def _connect_output(self, slot: Slot) -> List[Slot]:
        slot_list = self.get_output_slot(slot.output_name)
        if slot_list is None:
            slot_list = [slot]
            self._output_slots[slot.output_name] = slot_list
        else:
            slot_list.append(slot)
        return slot_list

    def _disconnect_output(self, name: str) -> None:
        slots = self._output_slots.get(name, None)
        if slots is None:
            logger.error("Cannot get output slot %s", name)
            return
        slots = [s for s in slots if s.output_name != name]
        self._output_slots[name] = slots
        # maybe del slots if it is empty and not required?

    def get_data(self, name: str) -> Any:
        if name == Module.TRACE_SLOT:
            return self.tracer.trace_stats()
        if name == Module.PARAMETERS_SLOT:
            return self._params
        return None

    @abstractmethod
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        """Run one step of the module, with a duration up to the 'howlong' parameter.

        Returns a dictionary with at least 5 pieces of information: 1)
        the new state among (ready, blocked, zombie),2) a number
        of read items, 3) a number of updated items (written), 4) a
        number of created items, and 5) the effective number of steps run.
        """
        raise NotImplementedError("run_step not defined")

    @staticmethod
    def next_state(slot: Slot) -> ModuleState:
        """Return state_ready if the slot has buffered information,
        or state_blocked otherwise.
        """
        if slot.has_buffered():
            return Module.state_ready
        return Module.state_blocked

    def _return_run_step(
        self, next_state: ModuleState, steps_run: int
    ) -> ReturnRunStep:
        assert next_state >= Module.state_ready and next_state <= Module.state_zombie
        self.steps_acc += steps_run
        return {"next_state": next_state, "steps_run": steps_run}

    def is_visualization(self) -> bool:
        return self.TAG_VISUALIZATION in self.tags

    def get_visualization(self) -> Optional[str]:
        return None

    def is_source(self) -> bool:
        return self.TAG_SOURCE in self.tags

    def is_greedy(self) -> bool:
        return self.TAG_GREEDY in self.tags

    def is_tagged(self, tag: str) -> bool:
        return tag in self.tags

    def is_created(self) -> bool:
        return self._state == Module.state_created

    def is_running(self) -> bool:
        return self._state == Module.state_running

    def prepare_run(self, run_number: int) -> None:
        "Switch from zombie to terminated, or update slots."
        if self.state == Module.state_zombie:
            self.state = Module.state_terminated
            return
        for slot in self.input_slot_values():
            if slot is None:
                continue
            slot.update(run_number)

    def is_ready(self) -> bool:
        if self.state == Module.state_terminated:
            logger.info("%s Not ready because it terminated", self.name)
            return False
        if self.state == Module.state_invalid:
            logger.info("%s Not ready because it is invalid", self.name)
            return False

        # Always process the input slots to have the buffers valid
        slots = self.input_slot_values()
        in_count = 0
        term_count = 0
        ready_count = 0
        for slot in slots:
            if slot is None:  # slot not required and not connected
                continue
            in_count += 1
            in_module = slot.output_module
            in_ts = in_module.last_update()
            ts = slot.last_update()

            if slot.has_buffered() or in_ts > ts:
                ready_count += 1
            elif in_module.is_terminated() or in_module.state == Module.state_invalid:
                term_count += 1

        logger.debug(
            f"ready_count={ready_count}, "
            f"term_count={term_count}, "
            f"in_count={in_count}"
        )
        if self.state == Module.state_blocked:
            # if all the input slot modules are terminated or invalid
            if (
                not self.is_input()  # input modules never die by themselves
                and in_count != 0
                and term_count == in_count
            ):
                logger.info(
                    "%s becomes zombie because all its input slots" " are terminated",
                    self.name,
                )
                self.state = Module.state_zombie
                return False
            # sources are always ready, and when 1 is ready, the module is.
            return in_count == 0 or ready_count != 0

        # Module is either a source or has buffered data to process
        if self.state == Module.state_ready:
            return True

        # source modules can be generators that
        # cannot run out of input, unless they decide so.
        if not self.has_any_input():
            return True

        # Module is waiting for some input, test if some is available
        # to let it run. If all the input modules are terminated,
        # the module is blocked, cannot run any more, so it is terminated
        # too.
        logger.error(
            "%s Not ready because is in weird state %s", self.name, self.state.name,
        )
        return False

    def cleanup_run(self, run_number: int) -> int:
        """Perform operations such as switching state from zombie to terminated.

        Resources could also be released for terminated modules.
        """
        if self.is_zombie():  # terminate modules that died in the previous run
            self.state = Module.state_terminated
        return run_number  # keep pylint happy

    def is_zombie(self) -> bool:
        return self._state == Module.state_zombie

    def is_terminated(self) -> bool:
        return self._state == Module.state_terminated

    def is_valid(self) -> bool:
        return self._state != Module.state_invalid

    @property
    def state(self) -> ModuleState:
        return self._state

    @state.setter
    def state(self, s: ModuleState) -> None:
        self.set_state(s)

    def set_state(self, s: ModuleState) -> None:
        assert (
            s.value >= Module.state_created.value
            and s.value <= Module.state_invalid.value
        ), "State %s invalid in module %s" % (s, self.name)
        self._state = s

    def trace_stats(self, max_runs: Optional[int] = None) -> Table:
        return self.tracer.trace_stats(max_runs)

    def predict_step_size(self, duration: float) -> int:
        self.predictor.fit(self.trace_stats())
        return self.predictor.predict(duration, self.default_step_size)

    def starting(self) -> None:
        pass

    def _stop(self, run_number: int) -> None:
        self._end_time = self._start_time
        self._last_update = run_number
        self._start_time = 0
        assert self.state != self.state_running
        self.end_run(run_number)

    def set_start_run(self, start_run: Optional[Callable[[Module, int], None]]) -> None:
        if start_run is None or callable(start_run):
            self._start_run = start_run
        else:
            raise ProgressiveError("value should be callable or None", start_run)

    def start_run(self, run_number: int) -> None:
        if self._start_run:
            self._start_run(self, run_number)

    def set_end_run(self, end_run: Optional[Callable[[Module, int], None]]) -> None:
        if end_run is None or callable(end_run):
            self._end_run = end_run
        else:
            raise ProgressiveError("value should be callable or None", end_run)

    def end_run(self, run_number: int) -> None:
        if self._end_run:
            self._end_run(self, run_number)

    def ending(self) -> None:
        """Ends a module.
        called when it is about the be removed from the scheduler
        """
        self._state = Module.state_terminated
        for islot in self._input_slots.values():
            if islot is not None:
                islot.reset()
        for oslots in self._output_slots.values():
            for oslot in oslots or []:
                if oslot is not None:
                    oslot.reset()

    def last_update(self) -> int:
        "Return the last time when the module was updated"
        return self._last_update

    def last_time(self) -> float:
        return self._end_time

    def _update_params(self, run_number: int) -> None:
        # pylint: disable=unused-argument
        pslot = self._input_slots[self.PARAMETERS_SLOT]
        if pslot is None or pslot.output_module is None:  # optional slot
            return
        df = pslot.data()
        if df is None:
            return
        raise NotImplementedError("Updating parameters not implemented yet")

    def current_params(self) -> Row:
        return cast(Row, self._params.last())

    def set_current_params(self, v: Dict[str, Any]) -> Dict[str, Any]:
        current = self.current_params()
        combined = dict(current)
        combined.update(v)
        self._params.add(combined)
        return v

    def has_input(self) -> bool:
        """Return True if the module received something via a from_input() call.
        Usually is a flag set by from_input() and deleted by the following
        run_step().
        See Variable module
        """
        return False

    def run(self, run_number: int) -> None:
        assert not self.is_running()
        self.steps_acc = 0
        next_state = self.state
        exception = None
        now = self.timer()
        quantum = self.scheduler().fix_quantum(self, self.params.quantum)
        tracer = self.tracer
        if quantum == 0:
            quantum = 0.1
            logger.error(
                "Quantum is 0 in %s, setting it to a" " reasonable value", self.name
            )
        self.state = Module.state_running
        self._start_time = now
        self._end_time = self._start_time + quantum
        self._update_params(run_number)

        run_step_ret = {}
        self.start_run(run_number)
        tracer.start_run(now, run_number)
        step_size = self.predict_step_size(quantum)
        logger.info(f"{self.name}: step_size={step_size}")
        if step_size != 0:
            # pylint: disable=broad-except
            try:
                tracer.before_run_step(now, run_number)
                if self.debug:
                    pdb.set_trace()
                run_step_ret = self.run_step(run_number, step_size, quantum)
                next_state = cast(ModuleState, run_step_ret["next_state"])
                now = self.timer()
            except ProgressiveStopIteration:
                logger.info("In Module.run(): Received a StopIteration")
                next_state = Module.state_zombie
                run_step_ret["next_state"] = next_state
                now = self.timer()
            except Exception as e:
                print_exc()
                next_state = Module.state_zombie
                run_step_ret["next_state"] = next_state
                now = self.timer()
                tracer.exception(now, run_number)
                exception = e
                self._had_error = True
                self._start_time = now
            finally:
                assert run_step_ret is not None, (
                    "Error: %s run_step_ret"
                    " not returning a dict" % self.pretty_typename()
                )
                if self.debug:
                    run_step_ret["debug"] = True
                tracer.after_run_step(now, run_number, **run_step_ret)
                self.state = next_state

            if self._start_time == 0 or self.state != Module.state_ready:
                tracer.run_stopped(now, run_number)
            self._start_time = now
        self.state = next_state
        if self.state == Module.state_zombie:
            tracer.terminated(now, run_number)
        progress = self.get_progress()
        tracer.end_run(
            now,
            run_number,
            progress_current=progress[0],
            progress_max=progress[1],
            quality=self.get_quality(),
        )
        self._stop(run_number)
        if exception:
            raise RuntimeError("{} {}".format(type(exception), exception))


class InputSlots:
    # pylint: disable=too-few-public-methods
    """
    Convenience class to refer to input slots by name
    as if they were attributes.
    """

    def __init__(self, module: Module):
        self.__dict__["module"] = module

    def __setattr__(self, name: Union[int, str], slot: Slot) -> None:
        assert isinstance(slot, Slot)
        assert slot.output_module is not None
        assert slot.output_name is not None
        slot.input_module = self.__dict__["module"]
        if isinstance(name, int):
            pos = name
            imod = slot.input_module
            assert imod is not None
            desc = [(k, sd.required) for (k, sd) in imod.input_descriptors.items()]
            assert pos < len(desc)
            name_, req = desc[pos]
            # pos slot and all slots before pos have to be "required"
            assert set([r for (_, r) in desc[: pos + 1]]) == set([True])
            slot.input_name = name_
        else:
            slot.input_name = name
        slot.connect()

    def __getattr__(self, name: str) -> Slot:
        raise ProgressiveError("Input slots cannot be read, only assigned to")

    def __getitem__(self, name: str) -> Slot:
        raise ProgressiveError("Input slots cannot be read, only assigned to")

    def __setitem__(self, name: Union[int, str, Tuple[str, Any]], slot: Slot) -> None:
        if isinstance(name, (int, str)):
            return self.__setattr__(name, slot)
        name, meta = name
        slot.meta = meta
        return self.__setattr__(name, slot)

    def __dir__(self) -> Iterable[str]:
        module: Module = self.__dict__["module"]
        return module.input_slot_names()


class OutputSlots:
    # pylint: disable=too-few-public-methods
    """
    Convenience class to refer to output slots by name
    as if they were attributes.
    """

    def __init__(self, module: Module):
        self.__dict__["module"] = module

    def __setattr__(self, name: str, slot: Slot) -> None:
        raise ProgressiveError("Output slots cannot be assigned, only read")

    def __getattr__(self, name: str) -> Slot:
        module: Module = self.__dict__["module"]
        return module.create_slot(name, None, None)

    def __getitem__(self, name: str) -> Slot:
        return self.__getattr__(name)

    def __dir__(self) -> Iterable[str]:
        module: Module = self.__dict__["module"]
        return module.output_slot_names()


def _print_len(x: Sized) -> None:
    if x is not None:
        print(len(x))


class Every(Module):
    "Module running a function at each iteration"
    inputs = [SlotDescriptor("df")]

    def __init__(
        self,
        proc: Callable[[Any], None] = _print_len,
        constant_time: bool = True,
        **kwds: Any,
    ) -> None:
        super(Every, self).__init__(**kwds)
        self._proc = proc
        self._constant_time = constant_time

    def predict_step_size(self, duration: float) -> int:
        if self._constant_time:
            return 1
        return super(Every, self).predict_step_size(duration)

    def run_step(
        self, run_number: int, step_size: float, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("df")
        df = slot.data()
        self._proc(df)
        slot.clear_buffers()
        return self._return_run_step(Module.state_blocked, steps_run=1)


def _prt(x: Any) -> None:
    print(x)


class Print(Every):
    "Module to print its input slot"

    def __init__(self, **kwds: Any) -> None:
        if "proc" not in kwds:
            kwds["proc"] = _prt
        super(Print, self).__init__(quantum=0.1, constant_time=True, **kwds)


def _islot_to_json(slot: Optional[Slot]) -> Optional[JSon]:
    if slot is None:
        return None
    return slot.to_json()


def _oslot_to_json(slots: Optional[List[Slot]]) -> Optional[List[Optional[JSon]]]:
    if slots is None:
        return None
    return [_islot_to_json(s) for s in slots]


def _create_table(tname: str, columns: Parameters) -> Table:
    dshape = ""
    data = {}
    for (name, dtype, val) in columns:
        if dshape:
            dshape += ","
        dshape += "%s: %s" % (name, dshape_from_dtype(dtype))
        data[name] = val
    dshape = "{" + dshape + "}"
    assert Group.default_internal
    table = Table(tname, dshape=dshape, storagegroup=Group.default_internal(tname))
    table.add(data)
    return table
