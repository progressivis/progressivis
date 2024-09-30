from __future__ import annotations

import copy
from typing import (
    Type,
    Any
)
from dataclasses import dataclass, field

from progressivis.core.module_facade import ModuleFacade, SlotProxy
from progressivis.core.api import Sink, Scheduler
from progressivis import Module, ProgressiveError
from progressivis.core.utils import get_random_name
from progressivis.stats import (
    Min,
    Max,
    KLLSketch,
    Var,
    Histogram1D,
    Histogram2D,
    Distinct,
    Sample
)
from progressivis.linalg import Log


@dataclass
class ModuleRegistry:
    output_name: str
    module_cls: Type[Module]
    module_kw: dict[str, Any] = field(default_factory=dict)
    module_hints: Any = None
    module_connect: dict[str, str] = field(default_factory=dict)


TABLE_REGISTRY: dict[str, ModuleRegistry] = {
    "min": ModuleRegistry("result", Min),
    "max": ModuleRegistry("result", Max),
    "percentiles": ModuleRegistry("result", KLLSketch),
    "var": ModuleRegistry("result", Var),
    "histogram": ModuleRegistry("result", Histogram1D),
    "histogram2d": ModuleRegistry("result", Histogram2D),
    "distinct": ModuleRegistry("result", Distinct, {}),
    "log": ModuleRegistry("result", Log, {}),
    "sample": ModuleRegistry("result", Sample, {}),
}


def table_register(name: str, output_name: str, module_cls: Type[Module]) -> None:
    if name in TABLE_REGISTRY:
        raise ProgressiveError(f"Name '{name}' already registered in TABLE_REGISTRY")
    TABLE_REGISTRY[name] = ModuleRegistry(output_name, module_cls)


class TableFacade(ModuleFacade):
    """
    Class implementing the ``ProgressiVis`` façades.

    .. warning::
        Do not instanciate this class directly!

    """
    registered_modules: dict[tuple[int, str], TableFacade] = {}

    @staticmethod
    def get_or_create(module: Module, table_slot: str) -> TableFacade:
        """
        Get or create a ``TableFacade`` encapsulating a Table API using slots
        that create modules on demand.
        Currently, we support the following API:
        min, max, percentiles, var, histogram, distinct

        Parameters
        ----------
        module:
            the module providing the data around which the facade will be built
        table_slot:
            the slot name on `module` (example: `result`)
        """
        tabmod = TableFacade.registered_modules.get((id(module), table_slot))
        if tabmod is None:
            tabmod = TableFacade(module, table_slot)
            TableFacade.registered_modules[
                (id(module), table_slot)
            ] = tabmod
            module.on_ending(lambda mod, _ : TableFacade.forget(mod))
        return tabmod

    @staticmethod
    def forget(module: Module, table_slot: str | None = None) -> None:
        """
        Remove a registered module class from the list of registered modules.

        Parameters
        ----------
        module:
            the module to be forgotten (if `table_slot_`is `None`) or
            containing the slot to be forgotten
        table_slot:
            the slot to be forgotten
        """
        mod_id = id(module)
        if table_slot is None:
            for k, v in TableFacade.registered_modules.items():
                if k[0] == mod_id:
                    del TableFacade.registered_modules[k]
        else:
            del TableFacade.registered_modules[(mod_id, table_slot)]

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
        if name in ("main", "result"):  # keeping result for module compatibility
            return SlotProxy(self.table_slot, self.module)
        scheduler = self.module.scheduler()
        reg = self.registry[name]
        mod = reg.module_cls(name=get_random_name(name), scheduler=scheduler, **reg.module_kw)
        mod.input[mod.default_input()] = self.module.output[self.table_slot][reg.module_hints]
        for k, v in reg.module_connect.items():
            mod.input[k] = self.output[v]
        # Add a sink to keep the module alive in case it is disconnected
        sink = Sink("sink_for_" + mod.name, scheduler=scheduler)
        sink.input.inp = mod.output[reg.output_name]
        return SlotProxy(reg.output_name, mod)  # TODO add the slot name in the registry

    def configure(self, *, base: str, hints: Any, name: str, connect: dict[str, str] = {}, **kw: Any) -> None:
        """
        Allows you to specialize a pre-existing descriptive module by adding elements for
        its instantiation or connection.

        Parameters
        ----------
        base:
            key associated to a pre-existing descriptive module (example ``max``)
        hints:
            information to connect to the main module (example: a sequence of columns)
        name:
            key associated to the new descriptive module
        kw:
            additional arguments for instanciation
        """
        if name in self.registry:
            raise ValueError(f"Name {name} already exists!")
        base_reg = self.registry[base]
        name_reg = copy.copy(base_reg)
        name_reg.module_hints = hints
        name_reg.module_connect = connect.copy()
        name_reg.module_kw = kw
        self.registry[name] = name_reg

    def scheduler(self) -> Scheduler:
        return self.module.scheduler()

    @property
    def members(self) -> list[str]:
        return list(self.registry.keys()) + ["main"]
