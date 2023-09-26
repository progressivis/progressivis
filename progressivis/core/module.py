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
from progressivis.table.table_base import BasePTable
from progressivis.table.table import PTable
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
from .slot_join import SlotJoin
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
    Coroutine,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .dataflow import Dataflow
    from .decorators import _Context

    Parameters = List[Tuple[str, np.dtype[Any], Any]]

PColumns = Union[None, List[str], Dict[str, List[str]]]
PCols = Union[None, List[str]]
PColsList = PCols
PColsDict = Union[None, Dict[str, List[str]]]
ModuleCb = Callable[["Module", int], None]
ModuleCoro = Callable[["Module", int], Coroutine[Any, Any, Any]]
ModuleProc = Union[ModuleCb, ModuleCoro]


JSon = Dict[str, Any]
# ReturnRunStep = Tuple[int, ModuleState]
ReturnRunStep = Dict[str, int]


logger = logging.getLogger(__name__)


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


class GroupContext:
    group: Optional[str] = None

    def __init__(self, modgroup: Union[str, Module]) -> None:
        self._saved = GroupContext.group
        GroupContext.group = modgroup if isinstance(modgroup, str) else modgroup.name

    def __enter__(self) -> GroupContext:
        return self

    def __exit__(self, *exc: Any) -> Any:
        GroupContext.group = self._saved
        return False


class ModuleState(IntEnum):
    state_created = 0
    state_ready = 1
    state_running = 2
    state_blocked = 3
    state_suspended = 4
    state_zombie = 5
    state_terminated = 6
    state_invalid = 7


class ModuleCallbackList(List[ModuleProc]):
    async def fire(self, module: Module, run_number: int) -> bool:
        ret = False
        for proc in self:
            try:
                if aio.iscoroutinefunction(proc):
                    coro = cast(ModuleCoro, proc)
                    await coro(module, run_number)
                else:
                    proc(module, run_number)
                ret = True
            except Exception as exc:
                logger.warning(exc)
        return ret


class Dependency:
    def __setattr__(self, name: str, value: Any) -> None:  # for mypy
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:  # for mypy
        return super().__getattr__(name)  # type: ignore


class Module(metaclass=ABCMeta):
    """
    Base Class for ``progressivis`` modules.

    .. warning::
        Do not instanciate this class directly!

    """

    parameters: Parameters = [
        ("quantum", np.dtype(float), 0.5),
        ("debug", np.dtype(bool), False),
    ]
    """ list of parameters managed by modules of this class """
    TRACE_SLOT = "_trace"
    PARAMETERS_SLOT = "_params"
    TAG_VISUALIZATION = "visualization"
    """ Tag attached to modules managing a visualization """
    TAG_INPUT = "input"
    """ Tag attached to input modules """
    TAG_SOURCE = "source"
    """ Tag attached to source modules """
    TAG_GREEDY = "greedy"
    """ Tag attached to greedy modules """
    TAG_DEPENDENT = "dependent"
    """ Tag attached to dependent modules """
    inputs = [SlotDescriptor(PARAMETERS_SLOT, type=BasePTable, required=False)]
    outputs = [SlotDescriptor(TRACE_SLOT, type=BasePTable, required=False)]
    output_attrs: Dict[str, str] = {}
    output_types: Dict[str, Any] = {}
    _doc_building: Optional[bool] = None
    state_created: ClassVar[ModuleState] = ModuleState.state_created
    state_ready: ClassVar[ModuleState] = ModuleState.state_ready
    state_running: ClassVar[ModuleState] = ModuleState.state_running
    state_blocked: ClassVar[ModuleState] = ModuleState.state_blocked
    state_suspended: ClassVar[ModuleState] = ModuleState.state_suspended
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
        columns: Optional[PColumns] = None,
        output_required: Optional[bool] = True,
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
        """ The module's name """
        predictor = TimePredictor.default()
        predictor.name = name
        self.predictor = predictor
        """ The Time Predictor used by this module """
        storage = StorageManager.default
        self.storage = storage
        if storagegroup is None:
            assert Group.default_internal is not None
            storagegroup = Group.default_internal(get_random_name(name + "_tracer"))
        self.storagegroup: Group = storagegroup
        tracer = Tracer.default(name, storagegroup)

        self.tags : Set[str] = set(ModuleTag.tags)
        """ The set oftags attached to this module """
        self.order: int = -1
        """ The order of this module in the scheduler, or -1 if not valid """
        self.group: Optional[str] = group or GroupContext.group
        """ The group this module belongs to """
        self.tracer = tracer
        self._start_time: float = 0
        self._end_time: float = 0
        self._last_update: int = 0
        self._state: ModuleState = Module.state_created
        self._saved_state: ModuleState = Module.state_invalid
        self._had_error = False
        self._parse_parameters(kwds)
        self.dep = Dependency()
        # always present
        input_descriptors = self.all_inputs
        output_descriptors = self.all_outputs
        self._input_slots: Dict[str, Optional[Slot]] = self._validate_descriptors(
            input_descriptors
        )
        self.input_descriptors: Dict[str, SlotDescriptor] = {
            d.name: d for d in input_descriptors
        }
        # self.input_multiple: Dict[str, int] = {
        #     d.name: 0 for d in input_descriptors if d.multiple
        # }
        self._output_slots: Dict[
            str, Optional[List[Slot]]
        ] = self._validate_descriptors(output_descriptors)
        self.output_descriptors: Dict[str, SlotDescriptor] = {
            d.name: d for d in output_descriptors
        }
        self.default_step_size: int = 100
        """ step size used by default when running this module for the first time """
        self.input = InputSlots(self)
        """ input slots are created and accessed through this write-only attribute """
        self.output = OutputSlots(self)
        """ output slots are created and accessed through this read-only attribute """
        self.steps_acc: int = 0
        # self.wait_expr = aio.FIRST_COMPLETED
        self.context: Optional[_Context] = None
        # callbacks
        self._start_run = ModuleCallbackList()
        self._after_run = ModuleCallbackList()
        self._ending: List[ModuleCb] = []
        # Register module
        dataflow.add_module(self)
        if "table_slot" in kwds:
            raise RuntimeError("don't use table_slot")
        if not output_required:
            # Change the descriptor so it's not required any more
            # The original SD is kept in the shared outputs/all_outputs
            # class variables
            sd = SlotDescriptor("result", type=PTable, required=False)
            self.output_descriptors["result"] = sd
        self._columns: Optional[List[str]] = None
        self._columns_dict: Dict[str, List[str]] = {}
        if isinstance(columns, dict):
            assert len(columns)
            for v in columns.values():
                self._columns = v
                break
            self._columns_dict = columns
        elif isinstance(columns, list):  # backward compatibility
            self._columns = columns
            for k in self._input_slots.keys():
                self._columns_dict[k] = columns
                break
        else:
            assert columns is None

    @staticmethod
    def tagged(*tags: str) -> ModuleTag:
        """Create a context manager to add tags to a set of modules
        created within a scope, typically dependent modules.
        """
        return ModuleTag(*tags)

    @staticmethod
    def doc_building() -> bool:
        if Module._doc_building is None:
            import sys

            if "sphinx.config" in sys.modules:
                print("doc building")
                Module._doc_building = True
            else:
                Module._doc_building = False
                print("no doc")
        return Module._doc_building
        return False

    @classmethod
    def finalize_doc(cls: Type[Module]) -> None:
        if not cls.doc_building():
            print("finalize: nothing to do on", cls)
            return
        if cls.__doc__ is None:
            cls.__doc__ = ""
        doclist = [cls.__doc__, "\n"]
        sd_keys = list(SlotDescriptor.__init__.__annotations__.keys())[
            1:
        ]  # exclude "name"
        defaults = SlotDescriptor.__init__.__defaults__
        assert defaults is not None
        sd_defs = dict(zip(sd_keys, defaults))
        print("sd_defs", sd_defs)

        def _section(n: int, s: str) -> None:
            sp = n * " "
            doclist.append(sp)
            doclist.append(s)
            doclist.append("\n")
            doclist.append(sp)
            doclist.append("_" * len(s))
            doclist.append("\n")

        def _param(n: int, name: str, typ: str, val: str) -> None:
            sp = n * " "
            sp2 = (n + 4) * " "
            typ = f"np.{t.name}" if type(t).__module__ == "numpy" else str(t)
            doclist.append(sp)
            doclist.append(f"{name}: {typ}\n")
            doclist.append(sp2)
            dv = '""' if val == "" else val
            doclist.append(f"default: {dv}\n")
            if name in params_doc:
                doclist.append(sp2)
                doclist.append(f", {params_doc[name]}\n")

        def _slot(
            n: int, sd: SlotDescriptor, cls: Type[Module], mode: str = "in"
        ) -> None:
            sp = n * " "
            sp2 = (n + 4) * " "
            doclist.append(sp)
            sd_type_name = (
                sd.type.__name__
                if sd.type is not None and hasattr(sd.type, "__name__")
                else "None"
            )
            doclist.append(f"{sd.name}: {sd_type_name}\n")
            no_defaults = []
            for k, v in sd_defs.items():
                if k == "type":
                    continue
                crt = getattr(sd, k)
                if crt != v:
                    no_defaults.append(f"{k}: {crt}")
            if mode == "out":
                attr_name = cls.output_attrs[sd.name]
                if attr_name != sd.name:
                    no_defaults.append(f"attr_name: {attr_name}")
                if attr_name not in cls.output_types:
                    no_defaults.append("custom_attr: True")
            if no_defaults:
                doclist.append(sp2)
                nd = ", ".join(no_defaults)
                doclist.append(f"{nd}\n")
            slot_doc = inputs_doc if mode == "in" else outputs_doc
            if sd.name in slot_doc:
                doclist.append(sp2)
                doclist.append(f" {slot_doc[sd.name]}\n")

        if cls.parameters:
            _section(4, "Module parameters")
            for n, t, v in cls.parameters:
                _param(8, n, str(t), v)
        if cls.inputs:
            _section(4, "Input slots")
            for sd in cls.inputs:
                if sd.name == "_params":
                    continue
                _slot(8, sd, cls)
        if cls.outputs:
            _section(4, "Output slots")
            for sd in cls.outputs:
                if sd.name == "_trace":
                    continue
                _slot(8, sd, cls, "out")
        raw_doc = "".join(doclist)
        from jinja2 import Template
        tmpl = Template(raw_doc)
        from progressivis_doc_params import napoleon_type_aliases  # type: ignore
        assert isinstance(napoleon_type_aliases, dict)
        cls.__doc__ = tmpl.render(**napoleon_type_aliases)

    def grouped(self) -> GroupContext:
        """Create a context manager to add group to a set of modules
        created within a scope, typically dependent modules.
        """
        return GroupContext(self.name)

    def scheduler(self) -> Scheduler:
        """Return the scheduler associated with the module."""
        return self._scheduler

    def dataflow(self) -> Optional[Dataflow]:
        """Return the dataflow associated with the module at creation time."""
        return self._scheduler.dataflow

    # def create_dependent_modules(self, *params, **kwds) -> None:  # pragma no cover
    #     """Create modules that this module depends on.
    #     """
    #     pass

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
        """Quality value, should increase when the quality increases."""
        return 0.0

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
        for name, _, _ in self.all_parameters:
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
        return sorted(
            [  # maintains the creation order
                iname for iname in self._input_slots if iname.startswith(prefix)
            ]
        )

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

    def reconnect(
        self, inputs: Dict[str, Slot], outputs: Dict[str, List[Slot]]
    ) -> None:
        logger.info(f"Module {self}, reconnect({inputs}, {outputs})")
        all_keys = set(self._input_slots.keys()) | set(inputs.keys())
        for name in all_keys:
            old_slot = self._input_slots.get(name, None)
            slot = inputs.get(name, None)
            if old_slot is slot:
                continue
            if old_slot is None:  # adding a new input slot
                # pylint: disable=protected-access
                assert slot is not None and slot.input_module is self
                # if slot.original_name:  # if it's a multiple slot, declare it
                #     descriptor = self.input_descriptors[slot.original_name]
                #     self.input_descriptors[name] = descriptor
                #     logger.info(
                #         'Creating multiple input slot "%s" in "%s"', name, self.name
                #     )
                self._input_slots[name] = slot
            else:
                # either delete a slot or replace an existing slot
                self._input_slots[name] = slot
                if slot is None:  # deleted slot
                    if old_slot.original_name:
                        # self.input_descriptors.pop(name)
                        logger.info(
                            f"Deleted multiple input slot {name} in {self.name}"
                        )
                    else:
                        logger.info(f"Deleted input slot {name} in {self.name}")
                else:  # slot is replaced
                    logger.info(f"Changing input slot {name} in {self.name}")
                    if (
                        slot == old_slot
                    ):  # no need to replace when the slots are identical
                        continue

        all_keys = set(self._output_slots.keys()) | set(outputs.keys())
        for name in all_keys:
            old_slots = self._output_slots.get(name, None)
            slots = outputs.get(name, None)
            if old_slots == slots:
                continue
            if old_slots is None:  # adding all new output slots
                logger.info(f"Module {self}, output '{name}': adding slots {slots}")
                self._output_slots[name] = slots
            elif slots is None:  # deleting all the output slots
                logger.info(
                    f"Module {self}: output '{name}': removing slots {old_slots}"
                )
                self._output_slots[name] = None
            else:
                new_slots: List[Slot] = []
                for slot in slots:
                    try:
                        index = old_slots.index(slot)
                    except ValueError:
                        index = -1
                    if index != -1:
                        old_slot = old_slots[index]
                        if slot == old_slot:  # if they are the same, reuse the old one
                            slot = old_slot
                    new_slots.append(slot)
                self._output_slots[name] = new_slots

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
            self.state = Module.state_ready

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
        output_attrs = type(self).output_attrs
        if name in output_attrs:
            return getattr(self, output_attrs[name])
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

    def _return_terminate(self, steps_run: int = 0) -> ReturnRunStep:
        return {"next_state": Module.state_zombie, "steps_run": steps_run}

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

    def is_suspended(self) -> bool:
        return self._state == Module.state_suspended

    def suspend(self) -> None:
        if self._state == Module.state_running:
            raise RuntimeError("Cannot suspend a running module (yet)")
        elif self._state == Module.state_suspended:
            return
        self._saved_state = self._state
        self._state = Module.state_suspended

    def resume(self) -> None:
        if self._state != Module.state_suspended:
            raise RuntimeError("Cannot resume a module not suspended")
        self._state = self._saved_state
        self._saved_state = Module.state_invalid

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
            logger.info(f"{self.name} Not ready because it terminated")
            return False
        if self.state == Module.state_invalid:
            logger.info(f"{self.name} Not ready because it is invalid")
            return False
        if self.state == Module.state_suspended:
            logger.info(f"{self.name} Not ready because it is suspended")
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
            "%s Not ready because is in weird state %s",
            self.name,
            self.state.name,
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

    def trace_stats(self, max_runs: Optional[int] = None) -> PTable:
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

    # TODO: rename to on_before_run
    def on_start_run(self, proc: ModuleProc) -> None:
        "Register a callback to call when the module starts to run"
        assert callable(proc)
        self._start_run.append(proc)

    def remove_start_run(self, proc: ModuleProc) -> None:
        "Remove the specified start_run callback"
        self._start_run.remove(proc)

    async def start_run(self, run_number: int) -> bool:
        return await self._start_run.fire(self, run_number)

    def on_after_run(self, proc: ModuleProc) -> None:
        "Register a callback to call when the module finishes to run"
        assert callable(proc)
        self._after_run.append(proc)

    def remove_after_run(self, proc: ModuleProc) -> None:
        "Remove the specified after_run callback"
        self._after_run.remove(proc)

    async def after_run(self, run_number: int) -> bool:
        return await self._after_run.fire(self, run_number)

    def on_ending(self, proc: ModuleCb) -> None:
        "Register a callback to call when the module ends"
        assert callable(proc) and not aio.iscoroutinefunction(proc)
        self._ending.append(proc)

    def remove_ending(self, proc: ModuleCb) -> None:
        "Remove the specified ending callback"
        self._ending.remove(proc)

    def do_ending(self, run_number: int) -> None:
        for proc in self._ending:
            try:
                proc(self, run_number)
            except Exception as exc:
                logger.warning("Exception in ending proc: ", exc)

    async def ending(self) -> None:
        """Ends a module.
        called when it is about the be removed from the scheduler
        """
        self.do_ending(self._last_update)
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
        "Return the current parameters"
        last = self._params.last()
        assert last is not None
        return last

    def set_current_params(self, v: Dict[str, Any]) -> Dict[str, Any]:
        "Change the current parameters"
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

    def _path_to_origin_impl(self) -> List[Module]:
        res = [
            item
            for sublist in [
                slot.output_module._path_to_origin_impl()
                for (sname, slot) in self._input_slots.items()
                if sname != "_params" and slot is not None
            ]
            for item in sublist
        ]
        res.append(self)
        return res

    def path_to_origin(self) -> Set[str]:
        lst = self._path_to_origin_impl()
        return set([m.name for m in lst])

    @property
    def all_parameters(self) -> Parameters:
        return self.parameters

    @property
    def all_inputs(self) -> List[SlotDescriptor]:
        return self.inputs

    @property
    def all_outputs(self) -> List[SlotDescriptor]:
        return self.outputs

    @property
    def columns(self) -> Optional[List[str]]:
        return (list(self._columns)
                if isinstance(self._columns, list)
                else self._columns)

    # Methods coming from TableModule

    def get_first_input_slot(self) -> Optional[str]:
        for k in self.input_slot_names():
            return k
        return None

    def get_columns(
        self, table: Union[BasePTable, Dict[str, Any]], slot: Optional[str] = None
    ) -> List[str]:
        """
        Return all the columns of interest from the specified table.

        If the module has been created without a list of columns, then all
        the columns of the table are returned.
        Otherwise, the interesection between the specified list and the
        existing columns is returned.
        """
        if table is None:
            return None
        if slot is None:
            _columns = self._columns
        else:
            _columns = self._columns_dict.get(slot)
        df_columns = table.keys() if isinstance(table, dict) else table.columns
        if _columns is None:
            _columns = list(df_columns)
        else:
            cols = set(_columns)
            diff = cols.difference(df_columns)
            for column in diff:
                _columns.remove(column)  # maintain the order
        return _columns

    def filter_columns(
        self,
        df: BasePTable,
        indices: Optional[Any] = None,
        slot: Optional[str] = None,
        cols: PColumns = None,
    ) -> BasePTable:
        """
        Return the specified table filtered by the specified indices and
        limited to the columns of interest.
        """
        if self._columns is None or (
            slot is not None and slot not in self._columns_dict
        ):
            if indices is None:
                return df
            return cast(BasePTable, df.loc[indices])
        cols = cols or self.get_columns(df, slot)
        if cols is None:
            return None
        if indices is None:
            if isinstance(cols, (int, str)):
                cols = slice(cols, cols)
            # return df[cols]
        return df.loc[indices, cols]  # type: ignore

    def get_slot_columns(self, name: str) -> List[str]:
        cols = self._columns_dict.get(name)
        if cols is not None:
            return cols
        slot = self.get_input_slot(name)
        if slot is not None:
            return list(slot.data().columns)
        return []

    def has_output_datashape(self, name: str = "table") -> bool:
        for osl in self.all_outputs:
            if osl.name == name:
                break
        else:
            raise ValueError("Output slot not declared")
        return osl.datashape is not None

    def get_output_datashape(self, name: str = "table") -> str:
        for osl in self.all_outputs:
            if osl.name == name:
                # output_ = osl
                break
        else:
            raise ValueError("Output slot not declared")
        if osl.datashape is None:
            raise ValueError("datashape not defined on {} output slot")
        dshapes = []
        for k, v in osl.datashape.items():
            isl = self.get_input_slot(k)
            assert isl is not None
            table = isl.data()
            if v == "#columns":
                colsn = self.get_slot_columns(k)
            elif v == "#all":
                colsn = table._columns
            else:
                assert isinstance(v, list)
                colsn = v
            for colname in colsn:
                col = table._column(colname)
                if len(col.shape) > 1:
                    dshapes.append(f"{col.name}: {col.shape[1]} * {col.dshape}")
                else:
                    dshapes.append(f"{col.name}: {col.dshape}")
        return "{" + ",".join(dshapes) + "}"

    def get_datashape_from_expr(self) -> str:
        if hasattr(self, "expr"):
            expr = getattr(self, "expr")
            return "{{{cols}}}".format(cols=",".join(expr.keys()))
        raise ValueError("expr attribute not defined")

    def make_slot_join(self, *slots: Slot) -> SlotJoin:
        return SlotJoin(self, *slots)

    def close_all(self) -> None:
        if self._params is not None and self._params.storagegroup is not None:
            self._params.storagegroup.close_all()
        for attr_name in self.output_attrs.values():
            outp = getattr(self, attr_name)
            if isinstance(outp, PTable) and outp.storagegroup is not None:
                outp.storagegroup.close_all()


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

    def __getattr__(self, name: Union[int, str]) -> Slot:
        module: Module = self.__dict__["module"]
        return module.create_slot(name, None, None)

    def __getitem__(self, name: Union[int, str]) -> Slot:
        return self.__getattr__(name)

    def __dir__(self) -> Iterable[str]:
        module: Module = self.__dict__["module"]
        return module.output_slot_names()


def _print_len(x: Sized) -> None:
    if x is not None:
        print(len(x))


params_doc = {}
inputs_doc = {}
outputs_doc = {}


def def_parameter(
    name: str, type: Any, value: Any, *, doc: str = ""
) -> Callable[[Type[Module]], Type[Module]]:
    """
    class decorator to declare a parameter

    Parameters
    ----------
    name:
        the parameter name
    type:
        the parameter type
    value:
        a default value for the parameter
    doc:
        an optional docstring
    """

    def module_decorator(module: Type[Module]) -> Type[Module]:
        par = (name, type, value)
        asdict = {p[0]: p for p in module.parameters}
        if name in asdict:
            asdict[name] = par
            module.parameters = list(asdict.values())
        else:
            module.parameters = [par] + module.parameters  # do not use += here
        if Module.doc_building() and doc:
            params_doc[name] = doc
        return module

    return module_decorator


def def_input(
    name: str, type: Any = None, *, doc: str = "", **kw: Any
) -> Callable[[Type[Module]], Type[Module]]:
    """
    class decorator to declare an input slot

    Parameters
    ----------
    name:
        the slot name
    type:
        the slot type
    doc:
        a docstring
    """

    def module_decorator(module: Type[Module]) -> Type[Module]:
        sd = SlotDescriptor(name=name, type=type, **kw)
        asdict = {s.name: s for s in module.inputs}
        if name in asdict:
            asdict[name] = sd
            module.inputs = list(asdict.values())
        else:
            module.inputs = [sd] + module.inputs  # do not use += here
        if Module.doc_building() and doc:
            inputs_doc[name] = doc
        return module

    return module_decorator


def _make_get(private_name: str) -> Callable[[Module], Any]:
    def result_get(obj: Module) -> Any:
        return getattr(obj, private_name)

    return result_get


def _make_set(
    private_name: str, stype: Type[type], module: Type[Module]
) -> Callable[[Module, Any], None]:
    def result_set(obj: Module, value: Any) -> None:
        if value is not None:
            if not isinstance(value, stype):
                raise ValueError(
                    f"{private_name}: {value} have to be an {stype} on {module.__name__}"
                )
        if getattr(obj, private_name) is not None:
            raise KeyError("result cannot be assigned more than once")
        setattr(obj, private_name, value)

    return result_set


def add_output_to_module(
    module: Type[Module],
    name: str,
    attr_name: str,
    custom_attr: bool,
    stype: Type[type],
) -> None:
    output_attrs = module.output_attrs
    if "output_attrs" not in module.__dict__:
        module.output_attrs = output_attrs.copy()
    module.output_attrs[name] = attr_name
    if custom_attr:
        return
    no_clash = "_progressivis"
    private_name = f"__{attr_name}_{no_clash}"
    setattr(module, private_name, None)
    result_get = _make_get(private_name)
    result_set = _make_set(private_name, stype, module)
    setattr(module, attr_name, property(result_get, result_set))
    # for mypy
    if "output_types" not in module.__dict__:
        module.output_types = {}
    module.output_types[attr_name] = stype


def def_output(
    name: str,
    type: Any = None,
    *,
    attr_name: Optional[str] = None,
    custom_attr: bool = False,
    doc: str = "",
    **kw: Any,
) -> Callable[[Type[Module]], Type[Module]]:
    """
    Class decorator to declare an output slot.
    An output is always associated with an underlying attribute on the current module.
    This attribute contains the value of the data of this slot from the module.
    By default this attribute is created by the decorator and it's name is the slot name.

    Parameters
    ----------
    name:
        the slot name
    type:
        the slot type
    attr_name:
        optional name for the slot underlying attribute. When missing, the attribute name
        is the slot name. This is useful, for example, to avoid a naming conflict.
    custom_attr:
        when ``True`` the underlying slot attribute is ``not`` created by the decorator.
        In this case the creation of the attribute is the responsibility of the module
        constructor. This is useful, for example, when the attribute requires special
        initialization.
    doc:
        a docstring
    """
    if attr_name is None:
        attr_name = name

    def module_decorator(module: Type[Module]) -> Type[Module]:
        sd = SlotDescriptor(name=name, type=type, **kw)
        asdict = {s.name: s for s in module.outputs}
        if name in asdict:
            asdict[name] = sd
            module.outputs = list(asdict.values())
        else:
            module.outputs = [sd] + module.outputs  # do not use += here
        assert attr_name is not None
        add_output_to_module(module, name, attr_name, custom_attr, type)
        if Module.doc_building() and doc:
            outputs_doc[name] = doc
        return module

    return module_decorator


def document(module: Type[Module]) -> Type[Module]:
    module.finalize_doc()
    print("params doc", module, params_doc)
    params_doc.clear()
    inputs_doc.clear()
    outputs_doc.clear()
    return module


@def_input("df")
class Every(Module):
    "Module running a function at each iteration"

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


def _create_table(tname: str, columns: Parameters) -> PTable:
    dshape = ""
    data = {}
    for name, dtype, val in columns:
        if dshape:
            dshape += ","
        dshape += "%s: %s" % (name, dshape_from_dtype(dtype))
        data[name] = val
    dshape = "{" + dshape + "}"
    assert Group.default_internal
    table = PTable(tname, dshape=dshape, storagegroup=Group.default_internal(tname))
    table.add(data)
    return table
