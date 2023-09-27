"""
Factory to manage modules associated with a source module

Examples:

factory = ModuleFactory()
ModuleFactory.register("Min", Min, "result")
ModuleFactory.register("Max", Max, "result")
ModuleFactory.register("RangeQuery", RangeQuery, "result")
ModuleFactory.register("RangeQuery2d", RangeQuery2d, "result")

"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

from progressivis.utils.errors import ProgressiveError
from progressivis.core import Scheduler
from progressivis.core import Module, ReturnRunStep
from progressivis.core import SlotDescriptor


class SlotDesc(NamedTuple):
    mod_name: str
    slot_name: str


class DerivedModule(NamedTuple):
    mod_type: type
    slot_name: str


class FactoryModule(Module):
    def __init__(
        self, scheduler: Scheduler, registry: Dict[str, DerivedModule]
    ) -> None:
        self.registry = registry
        inputs = []
        outputs = []
        # Prepare the slots before super
        for mod in registry.values():
            inputs.append(
                SlotDescriptor(name=mod.slot_name, type=mod.mod_type, required=False)
            )
            outputs.append(
                SlotDescriptor(name=mod.slot_name, type=mod.mod_type, required=False)
            )
        self.inputs = inputs + self.inputs
        self.outputs = outputs + self.outputs
        super(FactoryModule, self).__init__(scheduler=scheduler)

    def exists(self, name: str) -> bool:
        return name in self.registry

    def get_data(self, name: str) -> Any:
        return self.get_input_slot(name).data()

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        return self._return_run_step(Module.state_blocked, steps_run=1)


class ModuleFactory:
    def __init__(self) -> None:
        self.registry: Dict[str, DerivedModule] = {}

    def register(self, name: str, mod_type: type, slot_name: str) -> None:
        if name in self.registry:
            raise ProgressiveError("Module name already registered")
        self.registry[name] = DerivedModule(mod_type, slot_name)

    def make(self, scheduler: Optional[Scheduler] = None) -> FactoryModule:
        factory = FactoryModule(scheduler or Scheduler.default, self.registry)
        return factory
