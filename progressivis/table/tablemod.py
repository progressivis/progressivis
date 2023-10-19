from __future__ import annotations

from typing import (
    Type,
)
from dataclasses import dataclass

from progressivis.core.slot_hub import SlotHub, SlotProxy
from progressivis.core.sink import Sink
from progressivis import Module, ProgressiveError
from progressivis.core.utils import get_random_name
from progressivis.stats import (
    Min,
    Max,
    KLLSketch,
    Var,
    Histogram1D,
    Distinct,
)


@dataclass
class ModuleRegistry:
    output_name: str
    module_cls: Type[Module]


TABLE_REGISTRY: dict[str, ModuleRegistry] = {
    "min": ModuleRegistry("result", Min),
    "max": ModuleRegistry("result", Max),
    "percentiles": ModuleRegistry("result", KLLSketch),
    "var": ModuleRegistry("result", Var),
    "histogram": ModuleRegistry("result", Histogram1D),
    "distinct": ModuleRegistry("result", Distinct),
}


def table_register(name: str, output_name: str, module_cls: Type[Module]) -> None:
    if name in TABLE_REGISTRY:
        raise ProgressiveError(f"Name '{name}' already registered in TABLE_REGISTRY")
    TABLE_REGISTRY[name] = ModuleRegistry(output_name, module_cls)


class TableModule(SlotHub):
    def __init__(
        self,
        module: Module,
        table_slot: str,
        registry: dict[str, ModuleRegistry] | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.table_slot = table_slot
        self.registry: dict[str, ModuleRegistry] = registry or TABLE_REGISTRY
        # TODO: check it is an AbstractPTable?

    def unregister(self, name: str) -> None:
        del self.registry[name]

    def get(self, name: str) -> SlotProxy | None:
        return self._output_slots.get(name)

    def create_slot_module(self, name: str) -> SlotProxy:
        scheduler = self.module.scheduler()
        reg = self.registry[name]
        mod = reg.module_cls(name=get_random_name(name), scheduler=scheduler)
        mod.input.table = self.module.output[self.table_slot]
        sink = Sink("sink_for_" + mod.name, scheduler=scheduler)
        sink.input.inp = mod.output[reg.output_name]
        return SlotProxy(reg.output_name, mod)  # TODO add the slot name in the registry
