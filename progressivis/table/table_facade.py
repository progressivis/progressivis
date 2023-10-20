from __future__ import annotations

from typing import (
    Tuple,
    Type,
)
from dataclasses import dataclass

from progressivis.core.module_facade import ModuleFacade, SlotProxy
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


class TableFacade(ModuleFacade):
    registered_modules: dict[Tuple[str, str], TableFacade] = {}

    @staticmethod
    def get_or_create(module: Module, table_slot: str) -> TableFacade:
        """
        Get or create a TableFacade encapsulating a Table API using slots
        that create modules on demand. Currently, we support the following
        API:
        min, max, percentiles, var, histogram, distinct
        """
        tabmod = TableFacade.registered_modules.get((module.name, table_slot))
        if tabmod is None:
            tabmod = TableFacade(module, table_slot)
            TableFacade.registered_modules[
                (module.name, table_slot)
            ] = tabmod
            module.on_ending(lambda mod, _ : TableFacade.forget(mod))
        return tabmod

    @staticmethod
    def forget(module: Module, table_slot: str | None = None) -> None:
        """
        Remove a registered module class from the list of registered modules.
        """
        modname = module.name
        if table_slot is None:
            for k, v in TableFacade.registered_modules.items():
                if k[0] == modname:
                    del TableFacade.registered_modules[k]
        else:
            del TableFacade.registered_modules[(modname, table_slot)]

    def __init__(
        self,
        module: Module,
        table_slot: str,
        registry: dict[str, ModuleRegistry] | None = None,
    ) -> None:
        """Create a TableFacade. Don't use it directly, use the
        `get_or_create` static method instead.
        """
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
        # Add a sink to keep the module alive in case it is disconnected
        sink = Sink("sink_for_" + mod.name, scheduler=scheduler)
        sink.input.inp = mod.output[reg.output_name]
        return SlotProxy(reg.output_name, mod)  # TODO add the slot name in the registry
